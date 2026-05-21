"""LLM provider factory function."""

from typing import Optional

from .base import BaseLLM, LLMError
from .deepseek import DeepSeekLLM
from .kimi import KimiLLM
from .yunwu import YunwuLLM


def create_llm(provider: str = "deepseek", **kwargs) -> BaseLLM:
    """Create an LLM provider instance.

    Args:
        provider: One of 'deepseek', 'kimi', 'yunwu'.
        **kwargs: Passed through to the provider constructor.

    Returns:
        A BaseLLM instance.

    Raises:
        LLMError: If the provider is unknown or initialization fails.
    """
    provider = provider.lower().strip()
    if provider == "deepseek":
        return DeepSeekLLM(**kwargs)
    if provider in ("kimi", "kimi_k2"):
        return KimiLLM(**kwargs)
    if provider in ("yunwu", "yunwu_llm"):
        return YunwuLLM(**kwargs)
    raise LLMError(
        f"Unknown LLM provider: '{provider}'. Supported: deepseek, kimi, yunwu"
    )
