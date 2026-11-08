"""Local GRPO entry point; all reusable training logic lives in training.py."""
import argparse
import os
from dataclasses import fields

os.environ.setdefault("OMP_NUM_THREADS", "4")

from cs336_alignment.training import TrainingConfig, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    for field in fields(defaults):
        if field.name == "seed":
            continue
        default = getattr(defaults, field.name)
        kwargs = {"default": default}
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(default)
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = vars(parser.parse_args())
    seeds = args.pop("seeds")
    for seed in seeds:
        run_training(TrainingConfig(**args, seed=seed))


if __name__ == "__main__":
    main()
