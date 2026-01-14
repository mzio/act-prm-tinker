"""
Default replay buffer for storing training samples
"""

from typing import Any

from .types import EpisodeStep, Trajectory, TrajectoryGroup


class ReplayBuffer:
    """
    Replay buffer for storing state, action, next_obs, return, advantage
    """
    def __init__(self, max_size: int = 1e10) -> None:
        self.max_size = int(max_size)
        self.buffer: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []  # will convert these to torch.Tensor later
        # The only things we need to filter on are:
        # - split, batch_id, try_step, data_sample_id ?

    def add_episode_step(self, episode_step: EpisodeStep) -> None:
        """
        Add an episode step to the replay buffer
        """
        self.buffer.append({
            "state": episode_step.state,
            "action": episode_step.action,
            "next_obs": episode_step.next_obs,
            "tools": episode_step.tools,
            "old_logprobs": episode_step.old_logprobs,
            "return": episode_step.return_,
            "advantage": episode_step.advantage,
            # Metadata to filter on
            "split": episode_step.split,
            "batch_id": episode_step.batch_id,
            "try_step": episode_step.try_step,
            "data_sample_id": episode_step.unique_data_sample_id,
        })

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add all episode steps in a trajectory to the replay buffer
        """
        for episode_step in trajectory.episode_steps:
            self.add_episode_step(episode_step)

    def add_trajectory_group(self, trajectory_group: TrajectoryGroup) -> None:
        """
        Add all episode steps, in all trajectories in a trajectory group, to the replay buffer
        """
        for trajectory in trajectory_group.trajectories:
            self.add_trajectory(trajectory)
