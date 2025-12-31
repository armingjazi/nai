from base_provider import LLMProvider
from openrouter_provider import OpenRouterProvider
from together_provider import TogetherAIProvider
from openai_provider import OpenAIProvider
from anthropic_provider import AnthropicProvider


def get_provider(backend: str, model: str) -> LLMProvider:
    """Factory function to get the appropriate provider"""
    providers = {
        'openrouter': OpenRouterProvider,
        'together': TogetherAIProvider,
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
    }

    provider_class = providers.get(backend.lower())
    if not provider_class:
        raise ValueError(f"Unknown backend: {backend}. Available: {', '.join(providers.keys())}")

    return provider_class(model)
