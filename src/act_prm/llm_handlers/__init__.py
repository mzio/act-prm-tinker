"""
LLM classes and types
"""

from typing import Any

from .base import LLM
from .huggingface import HuggingFaceLLM
from .openai import (
    OpenAIResponsesLLM,
    AsyncOpenAIResponsesLLM,
    Response,
)
from .tinker import TinkerCompleter
from .types import ActionFromLLM


def load_llm(
    name: str,
    model_config: dict[str, Any],
    is_async: bool = False,
    **kwargs: Any,
) -> LLM:
    """
    Load LLM
    """
    if name == "hf_transformer":
        return HuggingFaceLLM(model_config=model_config, **kwargs)

    if name == "openai":
        if is_async:
            return AsyncOpenAIResponsesLLM(**model_config)
        else:
            return OpenAIResponsesLLM(**model_config)

    raise ValueError(f"Invalid model name: {name}")


__all__ = [
    "load_llm",
    "ActionFromLLM",
    "LLM",
    "Response",
    "OpenAIResponsesLLM",
    "AsyncOpenAIResponsesLLM",
    "HuggingFaceLLM",
    "TinkerCompleter",
]
