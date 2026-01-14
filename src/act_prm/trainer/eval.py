from typing import Any

import numpy as np
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

import tinker
from tinker_cookbook.renderers import Renderer

from act_prm.environments.base import Environment
from act_prm.generator.tinker import TinkerGenerator
from act_prm.llm_handlers.tinker import TinkerCompleter
from act_prm.replay_buffer.types import TrajectoryGroup
from .utils import gather_with_progress


async def run_evaluation(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    hf_tokenizer: PreTrainedTokenizerBase,
    eval_env: Environment,
    cfg: DictConfig,
    batch_id: int,
    split: str = "eval",
    num_tries: int = 1,
) -> dict[str, Any]:
    """
    Run evaluation for a single batch, e.g., by generating rollouts and grading them
    """
    tinker_completer = TinkerCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop_condition=cfg.stop_condition,
        hf_tokenizer=hf_tokenizer,
        **cfg.tool_call_kwargs,
    )
    tinker_generator = TinkerGenerator(
        llm=tinker_completer,
        env=eval_env,
        hf_tokenizer=hf_tokenizer,
        discount_factor=cfg.discount_factor,
        verbose=cfg.verbose,
    )

    all_eval_metrics = {}
    eval_metric_keys = [
        "final_reward", "first_return", "timesteps", "correct", "total",
    ]
    for try_idx in range(num_tries):
        all_trajectory_groups: list[list[TrajectoryGroup]] = await gather_with_progress(
            (
                tinker_generator.do_group_rollout(
                    num_return_sequences=cfg.eval_group_size,
                    split=split,
                    batch_id=batch_id,
                    unique_data_sample_id=sample_idx,
                    try_step=try_idx,
                    max_tokens=cfg.max_tokens,          # redundant as set in TinkerCompleter
                    temperature=cfg.temperature,        # redundant as set in TinkerCompleter
                )
                for sample_idx in range(len(eval_env))  # len(env) = number of tasks or problems
            ),
            desc=f"Evaluating {len(eval_env)} rollouts",
        )
        # Save metrics
        keys_for_correct = []
        for trajectory_group in all_trajectory_groups:
            for trajectory in trajectory_group:
                for metric_key in eval_metric_keys:
                    _metric_key = f"{split}/{try_idx}/{metric_key}"
                    if metric_key == "correct":
                        keys_for_correct.append(_metric_key)
                    if _metric_key not in all_eval_metrics:
                        all_eval_metrics[_metric_key] = []
                    val = getattr(trajectory, metric_key, 1)  # 1 for total samples
                    all_eval_metrics[_metric_key].append(val)

    # Compute aggregate metrics
    for k, v in all_eval_metrics.items():
        if "correct" in k or "total" in k:
            all_eval_metrics[k] = np.sum(v)
        else:
            all_eval_metrics[k] = np.mean(v)
        all_eval_metrics[f"{k}_std"] = np.std(v)
    # Add accuracy
    for k in keys_for_correct:
        total_v = all_eval_metrics[k.replace("correct", "total")]
        all_eval_metrics[k.replace("correct", "accuracy")] = all_eval_metrics[k] / total_v

    return all_eval_metrics
