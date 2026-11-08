"""Run the full local experiment matrix on four fixed training/inference GPU pairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

from cs336_alignment.training import TrainingConfig


def experiment_configs(seeds):
    configurations = []
    for seed in seeds:
        configurations.append(TrainingConfig(seed=seed))
    for lr in (3e-6, 3e-5):
        for seed in seeds:
            configurations.append(TrainingConfig(seed=seed, learning_rate=lr,
                                  output=f"results/grpo_learning_rate/lr_{str(lr).replace('e-0', 'e-')}"))
    for prompt in ("question_only", "r1_zero_three_shot"):
        for seed in seeds:
            configurations.append(TrainingConfig(seed=seed, prompt=prompt,
                                  output=f"results/grpo_prompt_ablation/{prompt}"))
    for variant in ("grpo_constant", "dr_grpo", "rft", "maxrl", "offpolicy_naive",
                    "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo", "loo"):
        for seed in seeds:
            configurations.append(TrainingConfig(seed=seed, variant=variant,
                                  output=f"results/grpo_{variant}"))
    return configurations


def gpu_pair_idle(pair):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True)
    memory = {int(line.split(",")[0]): int(line.split(",")[1]) for line in result.stdout.splitlines()}
    return all(memory[gpu] < 1000 for gpu in pair)


def run_worker(pair, pending):
    # One worker owns each physical pair for its entire lifetime.
    while True:
        while not gpu_pair_idle(pair):
            time.sleep(10)
        try:
            config = pending.get_nowait()
        except Empty:
            return
        output = Path(config.output) / f"seed_{config.seed}"
        try:
            if (output / "completed.json").exists():
                stored = json.loads((output / "config.json").read_text())["config"]
                for key in ("steps", "prompt", "learning_rate", "variant", "n_val", "rollout_batch_size"):
                    if stored.get(key, getattr(TrainingConfig(), key)) != getattr(config, key):
                        raise ValueError(f"Completed run configuration differs: {key}")
                print(f"SKIP {output}", flush=True)
                continue
            config.policy_device = f"cuda:{pair[0]}"
            config.inference_gpu = pair[1]
            config.port = 8000 + pair[0]
            config.resume = (output / "resume.pt").exists()
            output.mkdir(parents=True, exist_ok=True)
            if (output / "metrics.jsonl").exists() and not config.resume:
                raise RuntimeError("Existing incomplete run has no checkpoint; preserve it for manual inspection")
            command = [sys.executable, "scripts/train_grpo.py", "--seeds", str(config.seed)]
            for key, value in asdict(config).items():
                if key == "seed":
                    continue
                option = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    command.append(option if value else "--no-" + key.replace("_", "-"))
                else:
                    command.extend([option, str(value)])
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "4")
            # Physical IDs match CUDA IDs even if the invoking shell restricted visibility.
            env.pop("CUDA_VISIBLE_DEVICES", None)
            print(f"START pair={pair} {output}", flush=True)
            with (output / "process.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            if result.returncode:
                raise RuntimeError(f"Training exited {result.returncode}; see process.log for traceback/divergence")
            (output / "failure.json").unlink(missing_ok=True)
            print(f"DONE {output}", flush=True)
        except Exception as error:
            output.mkdir(parents=True, exist_ok=True)
            (output / "failure.json").write_text(json.dumps(dict(error=str(error), timestamp=time.time()), indent=2))
            print(f"FAILED {output}: {error}", flush=True)
        finally:
            pending.task_done()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--pairs", nargs="+", default=["0,1", "2,3", "4,5", "6,7"])
    parser.add_argument("--only", nargs="+", help="Select output directory names, e.g. grpo_standard grpo_rft")
    parser.add_argument("--exclude-run", nargs="+", default=[], help="Externally running seed directories to leave untouched")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pairs = [tuple(map(int, pair.split(","))) for pair in args.pairs]
    if any(len(pair) != 2 for pair in pairs) or len({gpu for pair in pairs for gpu in pair}) != 2 * len(pairs):
        raise ValueError("GPU pairs must have two distinct IDs and cannot overlap")
    configurations = [c for c in experiment_configs(args.seeds)
                      if (not args.only or Path(c.output).name in args.only)
                      and str(Path(c.output) / f"seed_{c.seed}") not in args.exclude_run]
    for excluded in args.exclude_run:
        print(f"EXTERNAL {excluded}: excluded; monitor this run separately", file=sys.stderr, flush=True)
    if args.dry_run:
        print(json.dumps([asdict(c) for c in configurations], indent=2))
        return
    pending = Queue()
    for config in configurations:
        pending.put(config)
    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        list(pool.map(lambda pair: run_worker(pair, pending), pairs))


if __name__ == "__main__":
    main()
