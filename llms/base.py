"""Abstract base class for LLM providers with Tool Calling support."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLM(ABC):
    """Abstract LLM interface supporting multi-turn conversation and optional Tool Calling."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send messages to the LLM and return the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of tool definitions in OpenAI function format.
            tool_choice: Optional tool selection strategy ('auto', 'none', etc.).

        Returns:
            Dict with keys:
              - 'content': str or None (text response)
              - 'tool_calls': list of tool call dicts or None
              - 'model': str (model identifier)
              - 'usage': dict or None (token counts)
        """

    def getResponse(self, message: str) -> str:
        """Simple single-message interface for backward compatibility.

        Args:
            message: A single user message string.

        Returns:
            The text content of the LLM's response.
        """
        result = self.chat([{"role": "user", "content": message}])
        return result.get("content", "")


class LLMError(Exception):
    """Raised when an LLM API call fails."""
    pass
