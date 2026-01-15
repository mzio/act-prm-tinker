"""
Tinker RL Trainer for fully synchronous on-policy training
"""

from os.path import join
from typing import Any, Callable
import logging
import time

from omegaconf import DictConfig

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import save_checkpoint_async
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from ..environments import Environment
from ..generator.tinker import TinkerGenerator
from ..replay_buffer import ReplayBuffer

from .tinker.utils import save_checkpoint_and_get_sampling_client, timed
from .train import is_better, run_rollouts, do_train_step_and_get_sampling_client

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Trainer for fully synchronous on-policy training with Tinker
    """
    def __init__(
        self, 
        cfg: DictConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        generator_constructor: Callable[..., TinkerGenerator],
        replay_buffer: ReplayBuffer,
        env: Environment,
        eval_env: Environment,  # could be the same as env, but update env.split
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.training_client = training_client
        self.service_client = service_client
        self.generator_constructor = generator_constructor
        self.replay_buffer = replay_buffer
        self.env = env
        self.eval_env = eval_env
        self.ml_logger = ml_logger
        self.hf_tokenizer = hf_tokenizer

        self.best_replay_buffer_path = join(cfg.checkpoint_path, "replay_buffer_best")
        self.last_replay_buffer_path = join(cfg.checkpoint_path, "replay_buffer")
        self.best_metric = 1e8 if cfg.best_metric in ["loss"] else -1e8
        self.best_sampling_client_path = ""

    # Modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L989
    async def train(self, start_batch: int, end_batch: int, cfg: DictConfig | None = None) -> None:
        """
        Implement fully synchronous on-policy training with Tinker

        For each training batch:
        1. Loads the most recent checkpoint and determines how we generate rollouts
        2. Generates rollouts (optionally running on the evaluation environment)
        3. Performs a policy update
        """
        cfg = cfg or self.cfg

        # Initial sampling client
        sampling_client, _ = await save_checkpoint_and_get_sampling_client(
            training_client=self.training_client,
            i_batch=start_batch,
            log_path=cfg.log_path,
            save_every=cfg.save_every,
            start_batch=start_batch,
        )

        model_name = cfg.model_name or self.training_client.get_info().model_data.model_name
        hf_tokenizer = self.hf_tokenizer or self.training_client.get_tokenizer()
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
                    self.eval_env.split = "eval"
                    eval_rollout_metrics, _, replay_buffer = await run_rollouts(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=self.generator_constructor,
                        replay_buffer=self.replay_buffer,
                        env=self.eval_env,
                        cfg=cfg,
                        batch_id=batch_idx,
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        start_idx=0,
                        tasks_per_update=len(self.eval_env),  # Just use all eval tasks
                    )
                    metrics.update(eval_rollout_metrics)

                # Save best checkpoints
                best_metric_key = f"eval/{cfg.eval_num_tries}/{cfg.best_metric}"
                last_metric = eval_rollout_metrics[best_metric_key]
                if is_better(last_metric, self.best_metric, cfg.best_metric):
                    self.best_metric = last_metric
                    self.replay_buffer.save_to_hf_dataset(self.best_replay_buffer_path)
                    path_dict = await save_checkpoint_async(
                        training_client=self.training_client,
                        name="best",
                        log_path=cfg.log_path,
                        loop_state={"batch": batch_idx},
                        kind="state",
                    )
                    self.best_sampling_client_path = path_dict["sampler_path"]
                    logger.info("Saved best replay buffer to %s", self.best_replay_buffer_path)
                    logger.info("Saved best sampling client to %s", self.best_sampling_client_path)
                    logger.info(
                        "Updated best %s to %f at batch %d",
                        cfg.best_metric, self.best_metric, batch_idx,
                    )

            # 1. Sample rollouts for training
            start_idx = batch_idx * cfg.batch_size
            tasks_per_update = cfg.batch_size
            self.env.split = "train"
            train_rollout_metrics, new_trajectories = await run_rollouts(
                sampling_client=sampling_client,
                renderer=renderer,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=self.generator_constructor,
                env=self.env,
                cfg=cfg,
                batch_id=batch_idx,
                split="train",
                num_tries=cfg.num_tries,
                start_idx=start_idx,
                tasks_per_update=tasks_per_update,
            )
            metrics.update(train_rollout_metrics)

            # Save replay buffer samples
            for trajectory in new_trajectories["policy"]:
                self.replay_buffer.add_trajectory(trajectory)
            self.replay_buffer.save_to_hf_dataset(self.last_replay_buffer_path)

            # 2. Update policy LLM with generated rollouts
            sampling_client, train_update_metrics = await do_train_step_and_get_sampling_client(
                cfg=cfg,
                batch_idx=batch_idx,
                training_client=self.training_client,
                service_client=self.service_client,
                new_trajectories=new_trajectories,
            )

            # Log metrics
            metrics.update(train_update_metrics)
            metrics["time/total"] = time.time() - t_start
            self.ml_logger.log_metrics(metrics, step=batch_idx)
