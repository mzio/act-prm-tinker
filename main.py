"""
Main script for training + evaluating LLMs

Example command:
```bash
uv run python main.py \
--is_async \
--env_config hotpotqa_mc/fewshot2 \
--generator_config default \
--trainer_config qwen3_4b_pg \
--replay_buffer_config default \
--log_path ./logs \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--seed 42 --replicate 0 --verbose
```
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils import ml_log

from act_prm.environments import get_env
from act_prm.generator import get_generator_constructor
from act_prm.replay_buffer import get_replay_buffer
from act_prm.setup import get_args, print_config, seed_everything
from act_prm.trainer.train import do_sync_training


logger = logging.getLogger(__name__)


def update_configs(args: argparse.Namespace, *configs: DictConfig) -> tuple[DictConfig, ...]:
    """
    Update configs with any specified + applicable command-line arguments
    """
    # A bit heinous, but loop through all configs to update any applicable args
    for config in configs:
        for argname, argval in vars(args).items():
            if argval is not None and argname in config:
                config[argname] = argval
    return configs


async def main() -> None:
    """
    Main training function
    """
    # Initialize experiment
    args = get_args()
    seed_everything(args.seed)
    load_dotenv()  # Setup environment variables from .env file

    # Get default configs
    env_cfg           = OmegaConf.load(f"./configs/environments/{args.env_config}.yaml")
    generator_cfg     = OmegaConf.load(f"./configs/generator/{args.generator_config}.yaml")
    trainer_cfg       = OmegaConf.load(f"./configs/trainer/{args.trainer_config}.yaml")
    replay_buffer_cfg = OmegaConf.load(f"./configs/replay_buffer/{args.replay_buffer_config}.yaml")
    
    if args.eval_env_config is not None:
        eval_env_cfg = OmegaConf.load(f"./configs/environments/{args.eval_env_config}.yaml")
    else:
        eval_env_cfg = env_cfg

    # Consolidate + update configs from args
    updated_cfgs = update_configs(
        args, env_cfg, eval_env_cfg, generator_cfg, trainer_cfg, replay_buffer_cfg,
    )
    if args.verbose:
        for cfg in updated_cfgs:
            print_config(cfg)
    env_cfg, eval_env_cfg, generator_cfg, trainer_cfg, replay_buffer_cfg = updated_cfgs
    cfg = trainer_cfg  # Main config to reference (has all Tinker training attributes)

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info("Resumed training from %s", resume_info["state_path"])
    elif cfg.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info("Loaded weights from %s", cfg.load_checkpoint_path)
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )

    # Get environment, replay buffer, and generator class
    env = get_env(**env_cfg)
    # Reuse env if eval_env not specified; we always specify the split for loading new tasks
    eval_env = get_env(**eval_env_cfg) if args.eval_env_config else env
    replay_buffer = get_replay_buffer(**replay_buffer_cfg)
    generator_ctor = get_generator_constructor(**generator_cfg, ml_logger=ml_logger)

    # Training loop
    num_batches = cfg.num_batches  # number of training steps
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        generator_constructor=generator_ctor,
        replay_buffer=replay_buffer,
        env=env,
        eval_env=eval_env,
        ml_logger=ml_logger,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            loop_state={"batch": num_batches},
            kind="both",
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
