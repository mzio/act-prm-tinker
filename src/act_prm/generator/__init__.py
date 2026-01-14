"""
Generators for LLM-based rollouts
"""

from functools import partial
from typing import Any, Callable

from .tinker import TinkerGenerator


def get_generator_constructor(name: str, **kwargs: Any) -> Callable[..., TinkerGenerator]:
    """
    Get a (partially initialized) TinkerGenerator constructor by name

    e.g., if **kwargs has all the necessary arguments (it won't in most cases),
    we can get the TinkerGenerator object via:

    ```python
    generator_ctor = get_generator_constructor(**generator_cfg)
    generator = generator_ctor()
    ```
    """
    if name == "tinker_default":
        return partial(TinkerGenerator, **kwargs)

    elif name == "tinker_grpo":
        from .tinker import TinkerGRPOGenerator
        return partial(TinkerGRPOGenerator, **kwargs)

    else:
        raise NotImplementedError(f"Sorry, generator {name} is not implemented yet.")

__all__ = [
    "get_generator_constructor",
    "TinkerGenerator",
]
