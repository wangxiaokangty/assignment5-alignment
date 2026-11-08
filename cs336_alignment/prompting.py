"""Local GSM8K prompting evaluation for Problem (prompting_baselines)."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ("question_only", "r1_zero", "r1_zero_three_shot")


def load_gsm8k(path):
    examples = []
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            row = json.loads(line)
            if "####" not in row["answer"]:
                raise ValueError("GSM8K answer is missing ####")
            examples.append(
                {
                    "question": row["question"],
                    "ground_truth": row["answer"].rsplit("####", 1)[1].strip(),
                }
            )
    return examples


def prompt_template(name):
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")
    filename = "r1_zero_three_shot_gsm8k" if name == "r1_zero_three_shot" else name
    return (ROOT / "cs336_alignment/prompts" / f"{filename}.prompt").read_text()


def sampling_parameters(name):
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")
    params = dict(temperature=1.0, top_p=1.0, max_tokens=512)
    if name != "question_only":
        params.update(stop=["</answer>"], include_stop_str_in_output=True)
    return params


def grade_response(name, response, ground_truth):
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")
    grader = question_only_reward_fn if name == "question_only" else r1_zero_reward_fn
    scores = grader(response, ground_truth)
    category = 1 if scores["answer_reward"] else (2 if scores["format_reward"] else 3)
    return {**scores, "category": category}


def summarize(records):
    counts = Counter(row["category"] for row in records)
    n = len(records)
    return dict(
        n=n,
        correct=counts[1],
        formatted_wrong=counts[2],
        unformatted=counts[3],
        accuracy=counts[1] / n if n else 0.0,
        format_rate=(counts[1] + counts[2]) / n if n else 0.0,
        truncated=sum(row["finish_reason"] == "length" for row in records),
    )


def load_results(directory=ROOT / "results/prompting_baselines"):
    directory = Path(directory)
    summaries = json.loads((directory / "summary.json").read_text())
    records = {
        name: [
            json.loads(line)
            for line in (directory / f"{name}.jsonl").read_text().split("\n")
            if line.strip()
        ]
        for name in PROMPTS
    }
    return summaries, records


def audit_examples(records, count=10):
    """Deterministic inspection sample, selected before manual review."""
    return [
        row
        for category in (2, 3)
        for row in [r for r in records if r["category"] == category][:count]
    ]


def run_evaluation(
    model, dataset, output, seed=0, limit=None, gpu_memory_utilization=0.5
):
    from vllm import LLM, SamplingParams
    import torch
    import vllm

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    examples = load_gsm8k(dataset)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        examples = examples[:limit]
    llm = LLM(
        model=str(Path(model).resolve()),
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=128,
        enforce_eager=True,
    )
    summary = {}
    for name in PROMPTS:
        template = prompt_template(name)
        prompts = [template.format(question=row["question"]) for row in examples]
        completions = llm.generate(prompts, SamplingParams(**sampling_parameters(name)))
        records = []
        for index, (example, prompt, completion) in enumerate(
            zip(examples, prompts, completions, strict=True)
        ):
            response = completion.outputs[0]
            records.append(
                dict(
                    index=index,
                    **example,
                    prompt=prompt,
                    response=response.text,
                    tokens=len(response.token_ids),
                    finish_reason=response.finish_reason,
                    **grade_response(name, response.text, example["ground_truth"]),
                )
            )
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        )
        summary[name] = summarize(records)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(name, summary[name], flush=True)
    metadata = dict(
        model=str(Path(model).resolve()),
        dataset=str(Path(dataset).resolve()),
        dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        seed=seed,
        limit=limit,
        torch=torch.__version__,
        vllm=vllm.__version__,
        gpu=torch.cuda.get_device_name(),
        dtype="bfloat16",
        sampling={name: sampling_parameters(name) for name in PROMPTS},
        prompt_sha256={
            name: hashlib.sha256(prompt_template(name).encode()).hexdigest()
            for name in PROMPTS
        },
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return summary
