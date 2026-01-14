"""
Setup utilities
"""
import os
import random
from argparse import Namespace

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed everything"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_run_name(
    args: Namespace,
    prefix: str = "",
    ignore_args: list[str] | None = None,
) -> str:
    """Return run name"""
    run_name = prefix
    ignore_args = ignore_args or []

    for argname, argval in vars(args).items():
        if argval is None or argname in ignore_args:
            continue
        argn = "".join([c[:2] for c in argname.split("_")])
        # Remove hyphens and dots, e.g., --model_name gpt-4.1-nano-2025-04-14
        argval = str(argval).replace("-", "_").replace(".", "_").replace("/", "_")
        run_name += f"-{argn}={argval}"

    # Add checkpoint and logging path identifiers if specified
    for argname in ["load_checkpoint_path"]:  # maybe include "log_path" if specified
        if getattr(args, argname, None) is not None:
            argn = "".join([c[0] for c in argname.split("_")])
            ckpt_id = "_".join(
                [
                    "=".join(["".join([x[0] for x in c.split("_")]) for c in s.split("=")])
                    for s in args.log_path.split("/")[-1].split("-")
                ]
            )
            run_name += f"-{argn}={ckpt_id}"

    # Last cleanups (brevity, ensure no unintended directory separators)
    run_name = run_name.replace("False", "0").replace("True", "1").replace("/", "_")
    return run_name
