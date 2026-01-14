"""
Replay buffer utilities
"""

def compute_returns_with_last_value(
    rewards: list[float],
    discount_factor: float = 0.9,
    dones: list[bool] | None = None,
    last_value: float = 0.0,
) -> list[float]:
    """
    Compute returns from rewards with last value
    - Applied over a single rollout
    """
    # returns = torch.zeros_like(rewards).float()  # shape is T
    returns: list[float] = [0.0 for _ in range(len(rewards))]
    _return = last_value
    for t in reversed(range(len(rewards))):
        mask = not dones[t] if dones is not None else 1
        _return = rewards[t] + discount_factor * _return * mask
        returns[t] = _return
    return returns
