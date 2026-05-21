import os
from openai import OpenAI
import json

from .deepseek import DeepSeekLLM
from .kimi import KimiLLM


class DeepSeek:
    """Backward-compatible wrapper for DeepSeekLLM.

    Usage:
        llm = DeepSeek()
        response = llm.getResponse("your message")
    """
    def __init__(self):
        self._impl = DeepSeekLLM()

    def getResponse(self, message):
        return self._impl.getResponse(message)


class Kimi_k2:
    """Backward-compatible wrapper for KimiLLM.

    Usage:
        llm = Kimi_k2()
        response = llm.getResponse("your message")
    """
    def __init__(self):
        self._impl = KimiLLM()

    def getResponse(self, message):
        return self._impl.getResponse(message)


# Test api connection
if __name__ == "__main__":
    message = ("Could you provide me with a piece of executable code that allows the particle swarm to escape its current local optimum?"
               "And Your response contains only code and is wrapped in `python`.")

    LLM = DeepSeek()
    code = LLM.getResponse(message)
    print(type(code))
