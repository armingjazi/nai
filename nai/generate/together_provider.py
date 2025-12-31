from typing import Optional, List, Dict, Any
import os
import json
import tempfile
import requests
from together import Together
from base_provider import LLMProvider


class TogetherAIProvider(LLMProvider):
    """Together AI API provider"""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.client = Together(api_key=self.api_key)

    def _get_api_key(self) -> str:
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment")
        return api_key

    def generate(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        response = requests.post(
            url="https://api.together.xyz/v1/chat/completions",
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

    def create_batch(self, inputs: List[Dict[str, Any]]) -> str:
        """Create batch job on Together AI"""
        # Create JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for input in inputs:
                json.dump(input, f)
                f.write('\n')
            temp_file = f.name

        # Upload file using Together SDK
        try:
            file_resp = self.client.files.upload(
                file=temp_file,
                purpose="batch-api",
                check=False
            )
            file_id = file_resp.id
            print(f"File uploaded successfully. File ID: {file_id}")
        except Exception as e:
            os.unlink(temp_file)
            raise Exception(f"File upload failed: {str(e)}")

        os.unlink(temp_file)

        # Create batch using Together SDK
        try:
            batch = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions"
            )
            # Together SDK v2 uses batch.job.id
            batch_id = batch.job.id if hasattr(batch, 'job') else batch.id
            print(f"Batch created successfully. Batch ID: {batch_id}")
            return batch_id
        except Exception as e:
            raise Exception(f"Batch creation failed: {str(e)}")

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch status from Together AI"""
        try:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status.upper() if hasattr(batch, 'status') else 'UNKNOWN'

            # Get output_file_id - different structure in SDK v1 vs v2
            output_file_id = None
            if hasattr(batch, 'output_file_id'):
                output_file_id = batch.output_file_id
            elif hasattr(batch, 'job') and hasattr(batch.job, 'output_file_id'):
                output_file_id = batch.job.output_file_id

            return {
                'status': status,
                'completed': status == 'COMPLETED',
                'failed': status == 'FAILED',
                'data': {'output_file_id': output_file_id, 'raw_batch': batch}
            }
        except Exception as e:
            raise Exception(f"Status check failed: {str(e)}")

    def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieve batch results from Together AI"""
        status = self.get_batch_status(batch_id)
        output_file_id = status['data'].get('output_file_id')

        if not output_file_id:
            raise Exception("No output file available")

        try:
            # Download file content using Together SDK
            # The SDK v2 uses .with_streaming_response.content()
            with self.client.files.with_streaming_response.content(id=output_file_id) as response:
                content = b''
                for chunk in response.iter_bytes():
                    content += chunk

                # Parse JSONL results
                results = []
                for line in content.decode('utf-8').strip().split('\n'):
                    if line:
                        results.append(json.loads(line))

                return results
        except Exception as e:
            raise Exception(f"Results download failed: {str(e)}")
