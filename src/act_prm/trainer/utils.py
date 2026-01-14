"""
Helper functions for Tinker training
"""

from typing import Any, Coroutine, Iterable, Sequence, TypeVar
import asyncio
import logging
import time
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm

import tinker
from tinker_cookbook import checkpoint_utils

logger = logging.getLogger(__name__)

T = TypeVar('T')


# Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/utils/misc_utils.py#L18
@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    """
    Update metrics with time taken for a given key
    """
    logger.info("Starting %s", key)
    tstart = time.time()
    yield
    logger.info("%s took %.2f seconds", key, time.time() - tstart)
    metrics[f"time/{key}"] = time.time() - tstart


# Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/utils/misc_utils.py#L58
def split_list(lst: Sequence[T], num_splits: int) -> list[list[T]]:
    """
    Split a sequence into a list of lists, where the sizes are as equal as possible,
    and the long and short lists are as uniformly distributed as possible.

    Args:
        lst: The sequence to split
        num_splits: Number of sublists to create

    Returns:
        A list of sublists with sizes differing by at most 1

    Raises:
        ValueError: If num_splits > len(lst) or num_splits <= 0

    Examples:
        >>> split_list([1, 2, 3, 4, 5], 2)
        [[1, 2, 3], [4, 5]]
        >>> split_list([1, 2, 3, 4, 5], 3)
        [[1, 2], [3, 4], [5]]
    """
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    if num_splits > len(lst):
        raise ValueError(f"Cannot split list of length {len(lst)} into {num_splits} parts")

    edges = np.linspace(0, len(lst), num_splits + 1).astype(int)
    return [list(lst[edges[i] : edges[i + 1]]) for i in range(num_splits)]


# Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L53
async def gather_with_progress(
    coroutines: Iterable[Coroutine[Any, Any, T]],
    desc: str,
    colour: str | None = None,
) -> list[T]:
    """
    Run coroutines concurrently with a progress bar that updates as each completes.

    This preserves the order of results (like asyncio.gather) while providing
    real-time progress feedback as individual coroutines complete.
    """
    coroutine_list = list(coroutines)
    pbar = tqdm(total=len(coroutine_list), desc=desc, colour=colour)

    async def track(coro: Coroutine[Any, Any, T]) -> T:
        result = await coro
        pbar.update(1)
        return result

    try:
        results = await asyncio.gather(*[track(coro) for coro in coroutine_list])
    finally:
        pbar.close()

    return results


# Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L714
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    Save checkpoint and get sampling client
    """
    metrics = {}
    with timed("save_checkpoint", metrics):
        if save_every > 0 and i_batch > start_batch and i_batch % save_every == 0:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{i_batch:06d}",
                log_path=log_path,
                loop_state={"batch": i_batch},
                kind="both",
            )
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        return await training_client.save_weights_and_get_sampling_client_async(), metrics
