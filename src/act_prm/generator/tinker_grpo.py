"""
Tinker Generator with Mean-Centered Advantages
"""

from typing import Any

from .tinker import TinkerGenerator
from ..replay_buffer.types import MeanCenteredTrajectoryGroup


class TinkerGRPOGenerator(TinkerGenerator):
    """
    Tinker Generator with Mean-Centered Return Rollouts
    """
    def _get_trajectory_group(self, **kwargs: Any) -> MeanCenteredTrajectoryGroup:
        """
        Returns trajectory group where we compute advantages by:
        1. Computing mean-centered final rewards: final_reward - mean(final_rewards)
        2. Optionally apply step-wise discounting to these values
        """
        return MeanCenteredTrajectoryGroup(**kwargs)
