"""Yunwu (云雾) third-party LLM provider with OpenAI-compatible API.

Supports models like gpt-5-thinking-all and gemini-3.1-pro-preview.
"""

import json
import os
from typing import Any, Dict, List, Optional

try:
    from .env_loader import load_env
    load_env()
except Exception:
    pass

from .base import BaseLLM, LLMError

YUNWU_DEFAULT_BASE_URL = "https://yuwwu.ai/v1"
YUNWU_DEFAULT_MODEL = "gpt-5-thinking-all"


class YunwuLLM(BaseLLM):
    """Yunwu (云雾) LLM provider — OpenAI-compatible third-party platform."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        from openai import OpenAI

        self._api_key = api_key or os.environ.get("YUNWU_API_KEY", "")
        if not self._api_key:
            raise LLMError("YUNWU_API_KEY not found in environment or .env")
        self.model = model or os.environ.get("YUNWU_AVAILABLE_MODEL", YUNWU_DEFAULT_MODEL)
        # If model is a list-like string, take the first one
        if isinstance(self.model, str) and self.model.startswith("["):
            import ast
            try:
                models = ast.literal_eval(self.model)
                self.model = models[0] if models else YUNWU_DEFAULT_MODEL
            except Exception:
                self.model = YUNWU_DEFAULT_MODEL

        self.base_url = base_url or os.environ.get("YUNWU_BASE_URL", YUNWU_DEFAULT_BASE_URL)
        self.system_prompt = system_prompt or "You are an expert in the field of particle swarm optimization."
        self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + list(messages)

        kwargs = {"model": self.model, "messages": messages}

        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"Yunwu API call failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise LLMError("Yunwu returned no choices")

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]

        return {
            "content": choice.message.content,
            "tool_calls": tool_calls,
            "model": response.model or self.model,
            "usage": (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            ),
        }

    @staticmethod
    def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            k: {
                                "type": v.get("type", "string"),
                                "description": v.get("description", ""),
                            }
                            for k, v in tool.get("parameters", {}).items()
                        },
                        "required": list(tool.get("parameters", {}).keys()),
                    },
                },
            })
        return openai_tools
