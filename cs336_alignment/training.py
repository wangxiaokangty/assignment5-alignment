"""Reproducible local on-policy GRPO runs, with JSONL metrics and generations."""

from dataclasses import asdict, dataclass
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.prompting import grade_response, load_gsm8k, prompt_template, sampling_parameters
from cs336_alignment.vllm_utils import VLLMServer


@dataclass
class TrainingConfig:
    model: str = "models/OLMo-2-0425-1B"
    train_data: str = "data/gsm8k/train.jsonl"
    val_data: str = "data/gsm8k/test.jsonl"
    output: str = "results/grpo_standard"
    prompt: str = "r1_zero"
    seed: int = 0
    steps: int = 200
    n_train: int = 6400
    n_val: int = 1024
    learning_rate: float = 1e-5
    rollout_batch_size: int = 256
    group_size: int = 8
    gradient_accumulation_steps: int = 32
    max_tokens: int = 512
    temperature: float = 1.0
    max_grad_norm: float = 1.0
    eval_every: int = 10
    log_rollouts_every: int = 40
    policy_device: str = "cuda:0"
    inference_gpu: int = 1  # Physical GPU ID, independent of trainer CUDA visibility.
    port: int = 8000
    save_checkpoint: bool = True
    resume: bool = False
    variant: str = "standard"
    train_batch_size: int = 8  # Used by off-policy runs; on-policy uses the rollout batch.


def append_jsonl(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def rollout_metrics(records):
    n = len(records)
    return {
        "reward": sum(row["reward"] for row in records) / n,
        "format_reward": sum(row["format_reward"] for row in records) / n,
        "answer_reward": sum(row["answer_reward"] for row in records) / n,
        "response_length": sum(row["tokens"] for row in records) / n,
        "truncation_rate": sum(row["finish_reason"] == "length" for row in records) / n,
    }


def sample_rollouts(server, examples, config, step, group_size):
    template = prompt_template(config.prompt)
    prompts = [template.format(question=row["question"]) for row in examples]
    params = sampling_parameters(config.prompt)
    params.update(n=group_size, seed=config.seed + step, max_tokens=config.max_tokens,
                  temperature=config.temperature)
    completions = server.generate_completions(prompts, params, batch_size=128)
    repeated = [(row, prompt) for row, prompt in zip(examples, prompts) for _ in range(group_size)]
    return [dict(**row, prompt=prompt, response=completion.text,
                 tokens=len(completion.token_ids), finish_reason=completion.finish_reason,
                 **grade_response(config.prompt, completion.text, row["ground_truth"]))
            for (row, prompt), completion in zip(repeated, completions, strict=True)]


def run_training(config: TrainingConfig):
    """Run one seed; step 0 is evaluated before any updates, final step after sync."""
    variant_settings(config)
    if config.variant.startswith("offpolicy_") and (
        config.train_batch_size % config.group_size or config.rollout_batch_size % config.train_batch_size
    ):
        raise ValueError("Off-policy minibatches must divide rollouts and preserve whole groups")

    if config.rollout_batch_size % config.group_size:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if config.rollout_batch_size % config.gradient_accumulation_steps:
        raise ValueError("rollout_batch_size must be divisible by gradient_accumulation_steps")
    if min(config.steps, config.n_val, config.n_train, config.group_size, config.eval_every, config.log_rollouts_every) < 1:
        raise ValueError("steps, validation size and logging intervals must be positive")
    output = Path(config.output) / f"seed_{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists() and not config.resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    if config.n_train < config.rollout_batch_size // config.group_size:
        raise ValueError("Training dataset must contain at least one prompt batch")
    if config.resume:
        if (output / "completed.json").exists():
            raise ValueError("Refusing to resume a completed run; its results remain unchanged")
        if not (output / "resume.pt").exists():
            raise FileNotFoundError(f"No recovery checkpoint in {output}")
        previous_metadata = json.loads((output / "config.json").read_text())
        previous = previous_metadata["config"]
        flexible = {"resume", "policy_device", "inference_gpu", "port", "save_checkpoint"}
        changed = [key for key, value in asdict(config).items()
                   if key not in flexible and previous.get(key, getattr(TrainingConfig(), key)) != value]
        if changed:
            raise ValueError(f"Resume configuration changed: {changed}")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    train = load_gsm8k(config.train_data)
    val = load_gsm8k(config.val_data)
    if len(train) < config.n_train or len(val) < config.n_val:
        raise ValueError("Requested more examples than dataset contains")
    train, val = train[:config.n_train], val[:config.n_val]
    metadata = dict(config=asdict(config), torch=torch.__version__,
                    train_sha256=hashlib.sha256(Path(config.train_data).read_bytes()).hexdigest(),
                    val_sha256=hashlib.sha256(Path(config.val_data).read_bytes()).hexdigest(),
                    prompt_sha256=hashlib.sha256(prompt_template(config.prompt).encode()).hexdigest())
    if config.resume:
        changed_inputs = [key for key in ("train_sha256", "val_sha256", "prompt_sha256")
                          if previous_metadata.get(key) != metadata[key]]
        if changed_inputs:
            raise ValueError(f"Resume input content changed: {changed_inputs}")
    else:
        (output / "config.json").write_text(json.dumps(metadata, indent=2))
    started = time.monotonic()
    server = VLLMServer(str(Path(config.model).resolve()), gpu=config.inference_gpu,
                        seed=config.seed, port=config.port, gpu_memory_utilization=0.65)
    try:
        server.start()
        model = AutoModelForCausalLM.from_pretrained(
            config.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(config.policy_device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      betas=(0.9, 0.95), weight_decay=0.0)
        start_step, previous_wall = 0, 0.0
        order, cursor = [], 0
        state_path = output / "resume.pt"
        if config.resume:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            rng.setstate(state["sampling_rng"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            start_step, previous_wall = state["step"], state["wall_seconds"]
            order, cursor = state["order"], state["cursor"]
            del state
            # Remove only observations newer than the recovered optimizer state.
            for name in ("metrics.jsonl", "rollouts.jsonl"):
                path = output / name
                if path.exists():
                    recovered = [row for row in read_recovery_jsonl(path)
                                 if row["step"] < start_step or
                                 (row["step"] == start_step and row["split"] == "train")]
                    temporary = path.with_suffix(".recovering")
                    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                 for row in recovered))
                    temporary.replace(path)
        started -= previous_wall
        server.init_weight_sync(config.policy_device)
        server.sync_policy_weights(model)
        for step in range(start_step, config.steps + 1):
            if step % config.eval_every == 0 or step == config.steps:
                records = sample_rollouts(server, val, config, step, 1)
                metrics = dict(step=step, split="val", wall_seconds=time.monotonic() - started,
                               **rollout_metrics(records))
                append_jsonl(output / "metrics.jsonl", metrics)
                print(json.dumps(metrics), flush=True)
                if step == 0 or step == config.steps or step % config.log_rollouts_every == 0:
                    for row in records:
                        append_jsonl(output / "rollouts.jsonl", dict(step=step, split="val", **row))
            if step == config.steps:
                break
            n_prompts = config.rollout_batch_size // config.group_size
            if cursor + n_prompts > len(order):
                order = rng.sample(range(len(train)), len(train))
                cursor = 0
            examples = [train[i] for i in order[cursor:cursor + n_prompts]]
            cursor += n_prompts
            rollout_started = time.monotonic()
            records = sample_rollouts(server, examples, config, step + 1, config.group_size)
            rollout_seconds = time.monotonic() - rollout_started
            update_started = time.monotonic()
            model.train()
            loss, stats = update_rollouts(model, tokenizer, optimizer, records, config)
            metrics = dict(step=step + 1, split="train", loss=float(loss), **stats,
                           **rollout_metrics(records), rollout_seconds=rollout_seconds,
                           update_seconds=time.monotonic() - update_started,
                           wall_seconds=time.monotonic() - started)
            append_jsonl(output / "metrics.jsonl", metrics)
            print(json.dumps(metrics), flush=True)
            if step == 0 or (step + 1) % config.log_rollouts_every == 0:
                for row in records:
                    append_jsonl(output / "rollouts.jsonl", dict(step=step + 1, split="train", **row))
            server.sync_policy_weights(model)
            if config.save_checkpoint and ((step + 1) % config.eval_every == 0 or step + 1 == config.steps):
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                sampling_rng=rng.getstate(), torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all(), step=step + 1,
                                order=order, cursor=cursor, wall_seconds=time.monotonic() - started),
                           output / "resume.tmp")
                (output / "resume.tmp").replace(output / "resume.pt")
        if config.save_checkpoint:
            model.save_pretrained(output / "checkpoint")
            tokenizer.save_pretrained(output / "checkpoint")
        (output / "completed.json").write_text(json.dumps(dict(steps=config.steps,
                                                wall_seconds=time.monotonic() - started)))
    finally:
        server.stop()
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def load_training_results(directory="results/grpo_standard"):
    """Read logged observations; incomplete runs remain explicitly identifiable."""
    directory = Path(directory)
    rows, runs = [], []
    for path in sorted(directory.glob("seed_*/config.json")):
        config = json.loads(path.read_text())["config"]
        complete = (path.parent / "completed.json").exists()
        runs.append(dict(seed=config["seed"], complete=complete, **{k: config[k] for k in
                         ("steps", "n_val", "rollout_batch_size", "max_tokens")}))
        metrics = path.parent / "metrics.jsonl"
        if metrics.exists():
            rows.extend(dict(seed=config["seed"], **json.loads(line))
                        for line in metrics.read_text().split("\n") if line.strip())
    return rows, runs


def plot_metric(rows, split, metric, output=None):
    """Plot each observed seed plus mean and min/max on shared steps."""
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r["split"] == split and metric in r
                and not (metric == "token_entropy" and r.get("active_fraction") == 0)]
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    seeds = sorted({r["seed"] for r in selected})
    steps = sorted({r["step"] for r in selected})
    for seed in seeds:
        data = sorted((r for r in selected if r["seed"] == seed), key=lambda r: r["step"])
        ax.plot([r["step"] for r in data], [r[metric] for r in data], alpha=.5, label=f"seed {seed}")
    values = {s: [r[metric] for r in selected if r["step"] == s] for s in steps}
    shared = [s for s in steps if len(values[s]) == len(seeds)]
    if shared:
        ax.fill_between(shared, [min(values[s]) for s in shared],
                        [max(values[s]) for s in shared], color="gray", alpha=.2, label="min–max")
        ax.plot(shared, [sum(values[s])/len(values[s]) for s in shared], color="black", label="mean")
        ax.legend()
    else:
        ax.text(.5, .5, "No observed data", transform=ax.transAxes, ha="center")
    ax.set(xlabel="Optimizer step", ylabel=metric, title=f"{split}/{metric}")
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def load_rollout_examples(directory="results/grpo_standard", seed=0, count=3):
    """Return matched first/last logged validation answers for qualitative review."""
    path = Path(directory) / f"seed_{seed}" / "rollouts.jsonl"
    if not path.exists():
        path = path.with_name("examples.jsonl")
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    validation = [r for r in records if r["split"] == "val"]
    if not validation:
        return []
    first, last = min(r["step"] for r in validation), max(r["step"] for r in validation)
    questions = list(dict.fromkeys(r["question"] for r in validation if r["step"] == first))[:count]
    return [{key: row[key] for key in ("step", "question", "ground_truth", "response", "reward", "format_reward")}
            for question in questions for row in validation
            if row["question"] == question and row["step"] in (first, last)]


VARIANTS = ("standard", "grpo_constant", "dr_grpo", "rft", "maxrl", "loo",
            "offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo")


def variant_settings(config):
    """Map assignment algorithms to the shared train-step interface."""
    if config.variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {config.variant}")
    settings = dict(baseline="mean", advantage_normalizer="std", loss_normalization="sequence",
                    importance_reweighting_method="none")
    if config.variant in ("grpo_constant", "dr_grpo", "rft", "maxrl", "loo"):
        settings.update(loss_normalization="constant",
                        normalization_constant=config.rollout_batch_size * config.max_tokens)
    if config.variant in ("dr_grpo", "rft", "loo"):
        settings["advantage_normalizer"] = "none"
    if config.variant == "rft":
        settings["baseline"] = "none"
    if config.variant == "loo":
        settings["baseline"] = "loo"
    if config.variant == "maxrl":
        settings["advantage_normalizer"] = "mean"
    method = {"offpolicy_noclip": "noclip", "offpolicy_clip": "grpo", "offpolicy_gspo": "gspo"}
    settings["importance_reweighting_method"] = method.get(config.variant, "none")
    if config.variant in ("offpolicy_clip", "offpolicy_gspo"):
        settings["cliprange"] = 0.2 if config.variant == "offpolicy_clip" else 3e-4
    return settings


@torch.no_grad()
def cache_rollout_log_probs(model, tokenizer, records, batch_size):
    """Score the entire rollout under the unchanged policy, caching on CPU.

    Global right padding and shifted labels are identical to train_step. Each
    later minibatch can safely trim the padded suffix without shifting tokens.
    """
    from cs336_alignment.policy import get_response_log_probs, tokenize_prompt_and_output
    batch = tokenize_prompt_and_output([r["prompt"] for r in records],
                                       [r["response"] for r in records], tokenizer)
    device = next(model.parameters()).device
    previous_mode = model.training
    model.eval()
    try:
        return torch.cat([
            get_response_log_probs(model, batch["input_ids"][start:start+batch_size].to(device),
                                   batch["labels"][start:start+batch_size].to(device))["log_probs"].cpu()
            for start in range(0, len(records), batch_size)
        ])
    finally:
        model.train(previous_mode)


def update_rollouts(model, tokenizer, optimizer, records, config):
    """One on-policy update or sequential whole-group off-policy minibatches.

    Returned training statistics are arithmetic means over optimizer calls,
    including inactive calls. Token entropy/clip fraction within each call are
    measured on active tokens, as documented by grpo_train_step.
    """
    from cs336_alignment.grpo import grpo_train_step
    settings = variant_settings(config)
    offpolicy = config.variant.startswith("offpolicy_")
    batch_size = config.train_batch_size if offpolicy else len(records)
    cache_started = time.monotonic()
    # Cache even for naive off-policy so timing comparisons include the same
    # scoring pass; that variant deliberately does not use importance ratios.
    old = cache_rollout_log_probs(model, tokenizer, records, batch_size) if offpolicy else None
    old_log_probs_seconds = time.monotonic() - cache_started
    losses, statistics = [], []
    for start in range(0, len(records), batch_size):
        minibatch = records[start:start+batch_size]
        loss, stats = grpo_train_step(
            model=model, tokenizer=tokenizer, optimizer=optimizer,
            gradient_accumulation_steps=1 if offpolicy else config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm, reward_fn=partial(grade_response, config.prompt),
            repeated_prompts=[r["prompt"] for r in minibatch],
            rollout_responses=[r["response"] for r in minibatch],
            repeated_ground_truths=[r["ground_truth"] for r in minibatch],
            group_size=config.group_size,
            old_log_probs=old[start:start+batch_size] if offpolicy else None, **settings,
        )
        losses.append(float(loss))
        statistics.append({key: float(value.detach().float().mean().cpu())
                           if isinstance(value, torch.Tensor) else float(value)
                           for key, value in stats.items()})
    averaged = {key: sum(s[key] for s in statistics) / len(statistics) for key in statistics[0]}
    averaged.update(old_log_probs_seconds=old_log_probs_seconds, optimizer_calls=len(statistics),
                    active_optimizer_calls=sum(s.get("active_fraction", 1) > 0 for s in statistics))
    if offpolicy:
        averaged.setdefault("clip_fraction", 0.0)
    return sum(losses) / len(losses), averaged



def read_recovery_jsonl(path):
    """Read a recovery log, dropping only a torn final JSON record.

    Earlier malformed records signal corruption and must never be hidden.
    """
    lines = [line for line in Path(path).read_bytes().split(b"\n") if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise
    return rows
