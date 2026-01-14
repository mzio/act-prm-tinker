"""
Parent environment class
References: https://github.com/bentrevett/pytorch-rl/blob/master/5%20-%20Proximal%20Policy%20Optimization%20(PPO)%20%5BCartPole%5D.ipynb
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from transformers import AutoTokenizer

from .types import EnvironmentState, EnvironmentStepResult

logger = logging.getLogger(__name__)

DEFAULT_TRUNCATION_TEMPLATE = (
    "Sorry, task not solved in max turns = {max_turns} allowed. Please try again."
)


class Environment(ABC):
    """
    Parent class for environment
    """

    def __init__(
        self,
        max_turns: int = 10000,
        num_tries: int = 1,
        tool_role: str = "tool",
        truncation_message: str = DEFAULT_TRUNCATION_TEMPLATE,
        split: str = "train",
        seed: int = 0,
        verbose: bool = False,
        pretrained_model_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.max_turns = max_turns
        self.num_tries = num_tries
        self.tool_role = tool_role
        self.truncation_message = truncation_message.format(max_turns=max_turns)
        self.split = split
        self.seed = seed
        self.verbose = verbose
        self.pretrained_model_config = pretrained_model_config
        self.tokenizer = self._init_tokenizer()

    def __len__(self) -> int:
        """
        Get the environment's number of sample tasks
        """
        return len(self.datasets[self.split])

    def adjust_sample_idx(self, sample_idx: int) -> int:
        """
        Adjust sample index to be in range of environment's number of tasks.
        -> Wrap around if out of bounds.
        """
        return sample_idx % len(self.datasets[self.split])

    def _init_tokenizer(self) -> AutoTokenizer | None:
        if self.pretrained_model_config is not None:
            _pretrained_model_config = {
                k: v for k, v in self.pretrained_model_config.items()
            }
            _model_name_or_path = _pretrained_model_config[
                "pretrained_model_name_or_path"
            ]
            if _model_name_or_path == "Qwen/Qwen3-8B":  # hack but get Qwen2.5 tokenizer
                _model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
            _pretrained_model_config["pretrained_model_name_or_path"] = (
                _model_name_or_path
            )
            _chat_template_path = _pretrained_model_config.pop("chat_template_path", None)
            tokenizer = AutoTokenizer.from_pretrained(**_pretrained_model_config)
            # Override chat template if provided
            if _chat_template_path is not None:
                with open(_chat_template_path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                    print(f"-> Overriding chat template with {_chat_template_path}")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        else:
            tokenizer = None
        return tokenizer

    @abstractmethod
    def reset(
        self,
        sample_idx: int,
        generation_idx: int,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> EnvironmentState:
        """
        Reset environment (starting new episode, or working on a new sample)
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, **kwargs: Any) -> EnvironmentStepResult:
        """
        Perform one step of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle environment's samples (e.g., after going through all during training)
        """
        raise NotImplementedError


class BaseTool(ABC):
    """
    Parent tool class
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """
        Execute the tool
        """
        raise NotImplementedError

    @abstractmethod
    def get_tool_desc(self) -> dict:
        """
        Returm the tool description for function calling
        """
        raise NotImplementedError
