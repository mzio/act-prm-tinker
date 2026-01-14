"""
Environments
"""

from typing import Any

from .base import Environment
from .types import EnvironmentState, EnvironmentStateWithAnswer


def get_env(name: str, is_async: bool = False, **kwargs: Any) -> Environment:
    """
    Get environment based on name
    """
    if name == "hotpotqa_mc":

        if is_async:
            from .hotpotqa_mc import AsyncHotpotQAMultipleChoiceEnv
            return AsyncHotpotQAMultipleChoiceEnv(**kwargs)
        else:
            from .hotpotqa_mc import HotpotQAMultipleChoiceEnv
            return HotpotQAMultipleChoiceEnv(**kwargs)

    elif name == "browsecomp_plus_search":
        if is_async:
            from .browsecomp_plus import AsyncBrowseCompPlusSearchEnv
            return AsyncBrowseCompPlusSearchEnv(**kwargs)
        else:
            from .browsecomp_plus import BrowseCompPlusSearchEnv
            return BrowseCompPlusSearchEnv(**kwargs)

    elif name == "longbench_v2":
        if is_async:
            from .longbench_v2 import AsyncLongBenchEnvironment
            return AsyncLongBenchEnvironment(**kwargs)
        else:
            from .longbench_v2 import LongBenchEnvironment
            return LongBenchEnvironment(**kwargs)

    raise NotImplementedError(f"Sorry invalid environment: '{name}'.")


def load_env(name: str, **kwargs: Any) -> Environment:
    """
    Alias for get_env
    """
    return get_env(name, **kwargs)


__all__ = [
    "get_env",
    "load_env",
    "Environment",
    "EnvironmentState",
    "EnvironmentStateWithAnswer",
]
