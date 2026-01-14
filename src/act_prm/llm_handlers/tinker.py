"""
Our version of Tinker completers

Copied and modified from tinker_cookbook/completers.py (v0.1.0)

Maybe this permalink?
https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/completers.py
"""

from dataclasses import dataclass
from typing import Any, TypeAlias

import tinker
from tinker_cookbook.renderers import Message, Renderer
from transformers import PreTrainedTokenizerBase

from .action_utils import get_actions, get_messages_from_text
from .types import ActionFromLLM

# Interfaces
StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    """
    Object to store tokens and logprobs for single model generation
    """
    tokens: list[int]
    maybe_logprobs: list[float] | None

    @property
    def logprobs(self) -> list[float]:
        """
        Retrieve logprobs from object
        """
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


@dataclass
class TokensWithLogprobsAndText(TokensWithLogprobs):
    """
    Object to store tokens, logprobs, and text content from single model generation
    """
    text: str
    is_complete: bool


class TinkerCompleter:
    """
    Base class for async Tinker completers (model generation)
    """
    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: Renderer,
        max_tokens: int,
        temperature: float = 1.0,
        stop_condition: StopCondition | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        tool_call_bos: str = "<tool_call>",
        tool_call_eos: str = "</tool_call>",
        tool_call_argname: str = "arguments",
    ) -> None:
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.hf_tokenizer = hf_tokenizer

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_condition = stop_condition or self.renderer.get_stop_sequences()

        self.tool_call_parse_kwargs: dict[str, str] = {
            "tool_call_bos": tool_call_bos,
            "tool_call_eos": tool_call_eos,
            "tool_call_argname": tool_call_argname,
        }

    async def __call__(
        self,
        **kwargs: Any,
    ) -> TokensWithLogprobs:
        """
        Generate tokens (see self.generate for class-specific implementation)
        """
        return await self.generate(**kwargs)

    async def generate(
        self,
        model_input: tinker.ModelInput,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: StopCondition | None = None,
    ) -> TokensWithLogprobs:
        """
        Generate tokens from model input
        """
        # Sample from model
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            stop=stop or self.stop_condition,
        )
        response = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )

        # Extract tokens and logprobs from the first (and only) sample
        sampled_tokens = response.sequences[0].tokens
        sampled_logprobs = response.sequences[0].logprobs
        assert sampled_logprobs is not None
        # Decode the response
        parsed_message, is_complete = self.renderer.parse_response(sampled_tokens)
        text_content = get_text_content(parsed_message)

        return TokensWithLogprobsAndText(
            tokens=sampled_tokens,
            maybe_logprobs=sampled_logprobs,
            text=text_content,
            is_complete=is_complete
        )

    async def get_actions(self, response: list[dict[str, Any]]) -> list[ActionFromLLM]:
        """
        Parse response into list of actions
        """
        return get_actions(response, **self.tool_call_parse_kwargs)

    async def get_messages_from_text(self, text: str) -> list[dict[str, Any]]:
        """
        Parse text into list of messages
        """
        return get_messages_from_text(text, **self.tool_call_parse_kwargs)

    async def compute_logprobs_async(self, model_input: tinker.ModelInput) -> list[float]:
        """
        Compute logprobs for a model input
        """
        return await self.sampling_client.compute_logprobs_async(model_input)


def get_text_content(message: Message, remove_thinking: bool = False) -> str:
    """
    Extract text content from message, optionally stripping thinking parts.
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    if remove_thinking:
        return "".join(p["text"] for p in content if p["type"] == "text")
    return "\n".join(p["text"] for p in content)
