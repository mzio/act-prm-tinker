"""
Replay buffer for storing episode steps (at minimum, (state, action, advantage) tuples)
"""

from typing import Any

from .base import ReplayBuffer


def get_replay_buffer(name: str, **kwargs: Any) -> ReplayBuffer:
    """
    Get a replay buffer by name
    """
    if name == "default":
        return ReplayBuffer(**kwargs)
    else:
        raise NotImplementedError(f"Sorry, replay buffer {name} is not implemented yet.")

__all__ = [
    "ReplayBuffer",
]
