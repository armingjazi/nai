from typing import Optional, List, Dict, Any
import os
import requests
from base_provider import LLMProvider


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider"""

    def _get_api_key(self) -> str:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        return api_key

    def generate(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
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
        raise NotImplementedError("OpenRouter does not support batch processing")

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        raise NotImplementedError("OpenRouter does not support batch processing")

    def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("OpenRouter does not support batch processing")
