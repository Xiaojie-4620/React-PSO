"""DeepSeek LLM provider with OpenAI-compatible Tool Calling."""

import json
import os
from typing import Any, Dict, List, Optional

try:
    from .env_loader import load_env
    load_env()
except Exception:
    pass

from .base import BaseLLM, LLMError

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"
DEEPSEEK_SYSTEM_PROMPT = "You are an expert in the field of particle swarm optimization."


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM provider using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        from openai import OpenAI

        self._api_key = api_key or os.environ.get("DeepseekApiKey", "")
        if not self._api_key:
            raise LLMError("DeepseekApiKey not found in environment")
        self.model = model or DEEPSEEK_DEFAULT_MODEL
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.system_prompt = system_prompt or DEEPSEEK_SYSTEM_PROMPT
        self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Prepend system message if not already present
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
            raise LLMError(f"DeepSeek API call failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise LLMError("DeepSeek returned no choices")

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
