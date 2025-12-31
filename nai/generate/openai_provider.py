from typing import Optional, List, Dict, Any
import os
import json
import tempfile
import requests
from base_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def _get_api_key(self) -> str:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return api_key

    def generate(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        response = requests.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"API Error: {response.status_code}, {response.text}")
            return None

    def create_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Create batch job on OpenAI"""
        # Create JSONL file (OpenAI format requires method and url fields)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for req in requests:
                # Transform to OpenAI batch format
                openai_req = {
                    "custom_id": req["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": req["body"]
                }
                json.dump(openai_req, f)
                f.write('\n')
            temp_file = f.name

        # Upload file
        with open(temp_file, 'rb') as f:
            response = requests.post(
                url="https://api.openai.com/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": f},
                data={"purpose": "batch"}
            )

        os.unlink(temp_file)

        if response.status_code != 200:
            raise Exception(f"File upload failed: {response.status_code}, {response.text}")

        file_id = response.json()['id']

        # Create batch
        response = requests.post(
            url="https://api.openai.com/v1/batches",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h"
            }
        )

        if response.status_code != 200:
            raise Exception(f"Batch creation failed: {response.status_code}, {response.text}")

        return response.json()['id']

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch status from OpenAI"""
        response = requests.get(
            url=f"https://api.openai.com/v1/batches/{batch_id}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )

        if response.status_code != 200:
            raise Exception(f"Status check failed: {response.status_code}, {response.text}")

        batch_data = response.json()
        status = batch_data.get('status', '')

        return {
            'status': status,
            'completed': status == 'completed',
            'failed': status in ['failed', 'expired', 'cancelled'],
            'data': batch_data
        }

    def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieve batch results from OpenAI"""
        status = self.get_batch_status(batch_id)
        output_file_id = status['data'].get('output_file_id')

        if not output_file_id:
            raise Exception("No output file available")

        response = requests.get(
            url=f"https://api.openai.com/v1/files/{output_file_id}/content",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Results download failed: {response.status_code}, {response.text}")

        # Parse JSONL results
        results = []
        for line in response.text.strip().split('\n'):
            if line:
                results.append(json.loads(line))

        return results
