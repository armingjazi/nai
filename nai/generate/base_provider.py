from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import time


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or self._get_api_key()

    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key from environment"""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        """Generate text from prompt (single request)"""
        pass

    @abstractmethod
    def create_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Create a batch job and return batch ID"""
        pass

    @abstractmethod
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch status"""
        pass

    @abstractmethod
    def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieve batch results"""
        pass

    def wait_for_batch(self, batch_id: str, poll_interval: int = 60) -> Dict[str, Any]:
        """Poll batch until completion"""
        print(f"Waiting for batch {batch_id} to complete...")
        while True:
            status = self.get_batch_status(batch_id)
            print(f"Batch status: {status.get('status', 'unknown')}")

            if status.get('completed', False):
                return status

            if status.get('failed', False):
                raise Exception(f"Batch failed: {status}")

            time.sleep(poll_interval)

    def process_batch(self, requests: List[Dict[str, Any]], poll_interval: int = 60) -> List[Dict[str, Any]]:
        """End-to-end batch processing"""
        batch_id = self.create_batch(requests)
        print(f"Created batch: {batch_id}")

        self.wait_for_batch(batch_id, poll_interval)
        print("Batch completed! Retrieving results...")

        return self.get_batch_results(batch_id)
