"""
Training and evaluation functions

Evaluation reference:
- https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/eval/custom_evaluators.py
- https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/eval/custom_evaluators.py#L1

Training reference (synchronous):
- https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/train.py
- https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L989

Dataset builder reference:
- https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/recipes/math_rl/math_env.py#L323

dataset is initialized where the len(dataset) is determined as the num_train_samples / batch_size
so start_batch = 0, end_batch = len(dataset) = num_train_samples / batch_size
  we then interate through these indices where dataset.get_batch(batch_idx) returns a batch of problems
"""

from typing import Any, Callable
import logging
import random
import time

import numpy as np
import torch

from omegaconf import DictConfig

import tinker
from tinker import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from ..environments import Environment
from ..generator.tinker import TinkerGenerator
from ..llm_handlers.tinker import TinkerCompleter
from ..replay_buffer import ReplayBuffer
from ..replay_buffer.types import Trajectory, TrajectoryGroup

from .metrics import incorporate_kl_penalty
from .policy_update import compute_full_batch_metrics_and_get_sampling_client, train_step
from .utils import (
    gather_with_progress,
    save_checkpoint_and_get_sampling_client,
    timed,
)

logger = logging.getLogger(__name__)


# -------------------
# Main training loops
# -------------------

async def do_sync_training(
    start_batch: int,  # starting batch or step = 1
    end_batch: int,    # ending batch, e.g., len(env) // cfg.batch_size
    cfg: DictConfig,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    generator_constructor: Callable[..., TinkerGenerator],
    replay_buffer: ReplayBuffer,
    env: Environment,
    eval_env: Environment,  # could be the same as env, but update env.split
    ml_logger: ml_log.Logger,
    hf_tokenizer: PreTrainedTokenizerBase | None = None,
) -> None:
    """
    Implement fully synchronous on-policy training with Tinker
    - Modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L989

    High-level logic:
    - For each training batch:
      1. Loads the most recent checkpoint and determines how we generate rollouts
      2. Generates rollouts (optionally running on the evaluation environment)
      3. Performs a policy update
    """
    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client=training_client,
        i_batch=start_batch,
        log_path=cfg.log_path,
        save_every=cfg.save_every,
        start_batch=start_batch,
    )

    model_name = cfg.model_name or training_client.get_info().model_data.model_name
    hf_tokenizer = hf_tokenizer or training_client.get_tokenizer()
    # ^Same as tinker_cookbook.tokenizer_utils.get_tokenizer(cfg.model_name)?
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, hf_tokenizer)
    logger.info("Using renderer: %s", renderer_name)

    num_batches = end_batch - start_batch
    for batch_idx in range(start_batch, end_batch):
        metrics = {
            "progress/batch": batch_idx,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (batch_idx + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and batch_idx % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                eval_env.split = "eval"
                eval_rollout_metrics, _, replay_buffer = await run_rollouts(
                    sampling_client=sampling_client,
                    renderer=renderer,
                    hf_tokenizer=hf_tokenizer,
                    generator_constructor=generator_constructor,
                    replay_buffer=replay_buffer,
                    env=eval_env,
                    cfg=cfg,
                    batch_id=batch_idx,
                    split="eval",
                    num_tries=cfg.eval_num_tries,
                    start_idx=0,
                    tasks_per_update=len(eval_env),  # Just use all eval tasks
                )
                metrics.update(eval_rollout_metrics)

        # 1. Sample rollouts for training
        start_idx = batch_idx * cfg.batch_size
        tasks_per_update = cfg.batch_size
        env.split = "train"
        train_rollout_metrics, new_trajectories, replay_buffer = await run_rollouts(
            sampling_client=sampling_client,
            renderer=renderer,
            hf_tokenizer=hf_tokenizer,
            generator_constructor=generator_constructor,
            replay_buffer=replay_buffer,
            env=env,
            cfg=cfg,
            batch_id=batch_idx,
            split="train",
            num_tries=cfg.num_tries,
            start_idx=start_idx,
            tasks_per_update=tasks_per_update,
        )
        metrics.update(train_rollout_metrics)

        # 2. Update policy LLM with generated rollouts
        sampling_client, train_update_metrics = await do_train_step_and_get_sampling_client(
            cfg=cfg,
            batch_idx=batch_idx,
            training_client=training_client,
            service_client=service_client,
            new_trajectories=new_trajectories,
        )

        # Log metrics
        metrics.update(train_update_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=batch_idx)


# -----------------------------
# Generation / Rollout Sampling
# -----------------------------

async def run_rollouts(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    hf_tokenizer: PreTrainedTokenizerBase,
    generator_constructor: Callable[..., TinkerGenerator],
    replay_buffer: ReplayBuffer,
    env: Environment,
    cfg: DictConfig,
    batch_id: int,
    split: str = "train",
    num_tries: int = 1,
    start_idx: int = 0,
    tasks_per_update: int | None = None,  # i.e., batch_size
) -> tuple[dict[str, Any], list[Trajectory], ReplayBuffer]:
    """
    Run rollouts for a single batch, e.g., by generating rollouts and grading them
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
    # Constructs a TinkerGenerator object
    tinker_generator = generator_constructor(
        llm=tinker_completer,
        env=env,
        hf_tokenizer=hf_tokenizer,
    )

    batch_size = tasks_per_update or len(env)  # len(env) is the number of tasks or problems
    all_eval_metrics = {}
    eval_metric_keys = [
        "final_reward", "first_return", "timesteps", "correct", "total",
    ]
    new_trajectories: list[Trajectory] = []
    for try_idx in range(num_tries):
        all_trajectory_groups: list[list[TrajectoryGroup]] = await gather_with_progress(
            (
                tinker_generator.do_group_rollout(
                    num_return_sequences=cfg.group_size if split == "train" else cfg.eval_group_size,
                    split=split,
                    batch_id=batch_id,
                    unique_data_sample_id=sample_idx,
                    try_step=try_idx,
                    max_tokens=cfg.max_tokens,   # redundant as set in TinkerCompleter
                    temperature=cfg.temperature, # redundant as set in TinkerCompleter
                )
                for sample_idx in range(start_idx, start_idx + batch_size)  
            ),
            desc=f"Generating {batch_size} rollouts ({split} split)",
        )
        # Save metrics and samples
        keys_for_correct = []
        for trajectory_groups in all_trajectory_groups:     # list of list of trajectory groups
            for traj_group in trajectory_groups:            # len(all_trajectory_groups) usually 1,
                for trajectory in traj_group.trajectories:  # but may be >1, e.g., if step-wise adv
                    for metric_key in eval_metric_keys:
                        _metric_key = f"{split}/{try_idx}/{metric_key}"
                        if metric_key == "correct":
                            keys_for_correct.append(_metric_key)
                        if _metric_key not in all_eval_metrics:
                            all_eval_metrics[_metric_key] = []
                        val = getattr(trajectory, metric_key, 1)  # 1 for total samples
                        all_eval_metrics[_metric_key].append(val)
                    # Add trajectory to list of new trajectories
                    new_trajectories.append(trajectory)
                    # Also add trajectory to replay buffer (saves all episode steps to the buffer)
                    replay_buffer.add_trajectory(trajectory)

    final_metrics = {}  # return these metrics for the batch
    # 1. Compute aggregate metrics
    for k, v in all_eval_metrics.items():
        if "correct" in k or "total" in k:
            final_metrics[k] = np.sum(v).item()  # convert to float for json.dumps
        else:
            final_metrics[k] = np.mean(v).item()
        final_metrics[f"{k}_std"] = np.std(v).item()
    # 2. Add accuracy
    for k in keys_for_correct:
        total_v = final_metrics[k.replace("correct", "total")]
        final_metrics[k.replace("correct", "accuracy")] = final_metrics[k] / total_v

    return final_metrics, new_trajectories, replay_buffer


# --------------------------
# Training / Policy Updating
# --------------------------

async def prepare_minibatch(
    new_trajectories: list[Trajectory],
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """
    Prepare a minibatch of trajectories for training
    """
    metrics = {}
    # Assemble training data
    with timed("assemble_training_data", metrics):
        data_D: list[tinker.Datum] = []
        metadata_D: list[dict[str, int]] = []
        for trajectory in new_trajectories:
            for episode_step in trajectory.episode_steps:
                sa_input_ids = episode_step.state_action_tokens
                act_logprobs = episode_step.old_logprobs
                input_tokens = sa_input_ids[:-1]
                target_tokens = sa_input_ids[1:]
                target_state_len = episode_step.state_len - 1

                padded_logprobs = [0.0] * target_state_len + act_logprobs
                adv = episode_step.advantage
                padded_advantages = [0.0] * target_state_len + [adv] * len(act_logprobs)
                padded_mask = [0.0] * target_state_len + [1.0] * len(act_logprobs)

                assert (
                    len(input_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                    == len(target_tokens)
                )
                metadata_D.append({
                    "sample_id": episode_step.unique_data_sample_id,
                    "generation_id": episode_step.generation_id,
                })
                data_D.append(
                    tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                            "mask": TensorData.from_torch(torch.tensor(padded_mask)),  # for KL
                        },
                    )
                )
    # Incorporate KL penalty if configured
    # - Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L763
    if kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=model_name),
                # ^^^ TODO: replace with the model we load, if relevant
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


# Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L941
async def do_train_step_and_get_sampling_client(
    cfg: DictConfig,
    batch_idx: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    new_trajectories: list[Trajectory],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    Update LLM policy with new trajectories and return updated sampling client
    """
    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        new_trajectories,
        service_client,
        model_name=cfg.model_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
    )
    metrics.update(prepare_minibatch_metrics)

    # Randomly subsample training data if not evenly divisible by num_substeps (# of mini-batches)
    # -> Tinker requires this: https://tinker-docs.thinkingmachines.ai/rl/rl-hyperparams#multiple-updates-per-sampling-iteration
    if len(data_D) % cfg.num_substeps != 0:
        new_batch_size = len(data_D) // cfg.num_substeps * cfg.num_substeps
        data_D = random.sample(data_D, new_batch_size)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )
    
    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        batch_idx + 1,  # NOTE: saving the checkpoint as the i + 1 step
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics
