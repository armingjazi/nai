from typing import Optional, List, Dict, Any
import os
import json
import requests
from base_provider import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic API provider"""

    def _get_api_key(self) -> str:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        return api_key

    def generate(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        response = requests.post(
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
        )

        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            print(f"API Error: {response.status_code}, {response.text}")
            return None

    def create_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Create batch job on Anthropic"""
        # Transform to Anthropic batch format
        anthropic_requests = []
        for req in requests:
            anthropic_requests.append({
                "custom_id": req["custom_id"],
                "params": req["body"]
            })

        response = requests.post(
            url="https://api.anthropic.com/v1/messages/batches",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={"requests": anthropic_requests}
        )

        if response.status_code != 200:
            raise Exception(f"Batch creation failed: {response.status_code}, {response.text}")

        return response.json()['id']

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch status from Anthropic"""
        response = requests.get(
            url=f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
        )

        if response.status_code != 200:
            raise Exception(f"Status check failed: {response.status_code}, {response.text}")

        batch_data = response.json()
        status = batch_data.get('processing_status', '')

        return {
            'status': status,
            'completed': status == 'ended',
            'failed': False,  # Anthropic doesn't have a failed status, individual requests can fail
            'data': batch_data
        }

    def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieve batch results from Anthropic"""
        status = self.get_batch_status(batch_id)
        results_url = status['data'].get('results_url')

        if not results_url:
            raise Exception("No results URL available")

        response = requests.get(
            url=results_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            },
            stream=True
        )

        if response.status_code != 200:
            raise Exception(f"Results download failed: {response.status_code}, {response.text}")

        # Parse JSONL results (streamed)
        results = []
        for line in response.iter_lines():
            if line:
                results.append(json.loads(line.decode('utf-8')))

        return results
