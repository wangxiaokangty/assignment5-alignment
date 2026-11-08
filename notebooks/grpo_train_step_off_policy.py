import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cs336_alignment.grpo import grpo_train_step
    from cs336_alignment.policy import get_response_log_probs, tokenize_prompt_and_output
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    return AutoModelForCausalLM, AutoTokenizer, Path, get_response_log_probs, grpo_train_step, mo, r1_zero_reward_fn, tokenize_prompt_and_output, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — grpo_train_step_off_policy

    对 rollout policy 的 log probabilities 做 detach，并在各次训练更新中保持固定。
    剪除零 advantage 后按原始行号选取 old log probabilities，再裁掉右侧多余 padding，
    从而保持 old/new token 对齐。支持不裁剪、GRPO token ratio 和 GSPO sequence ratio。
    `clip_fraction` 按 active response token 数加权汇总，表示裁剪实际限制目标的 token 比例；
    熵同样只覆盖 active response tokens，`active_fraction` 给出保留回复比例。

    以下用本地 OLMo 对固定示例缓存一次 old log probabilities，再执行连续更新。
    每种方法执行两次，因此后续更新相对于缓存策略是 off-policy。
    特意令零奖励回复最长，以实际验证剪除后 old/new padding 宽度不同时的对齐。
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, Path, torch):
    model_path = Path(__file__).resolve().parents[1] / "models/OLMo-2-0425-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=True, attn_implementation="sdpa",
    ).to(device).eval()
    model.config.use_cache = False
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    return device, model, optimizer, tokenizer


@app.cell
def _(device, get_response_log_probs, model, tokenize_prompt_and_output, tokenizer, torch):
    prompts = ["Question: What is 2 + 2?\nAnswer:"] * 4
    responses = [
        "<think>Add.</think> <answer>4</answer>",
        "<think>This is an intentionally much longer incorrect explanation.</think> <answer>5</answer>",
        "<think>Add two.</think> <answer>4</answer>",
        "<think>Guess.</think> <answer>3</answer>",
    ]
    batch = {key: value.to(device) for key, value in tokenize_prompt_and_output(
        prompts, responses, tokenizer
    ).items()}
    with torch.no_grad():
        old_log_probs = get_response_log_probs(model, batch["input_ids"], batch["labels"])["log_probs"]
    old_log_probs.requires_grad_(True)
    active_batch = tokenize_prompt_and_output(prompts[::2], responses[::2], tokenizer)
    assert active_batch["input_ids"].shape[1] < old_log_probs.shape[1]
    return old_log_probs, prompts, responses


@app.cell
def _(grpo_train_step, model, old_log_probs, optimizer, prompts, r1_zero_reward_fn, responses, tokenizer, torch):
    rows = []
    for method in ("noclip", "grpo", "gspo"):
        for update in range(2):
            loss, metadata = grpo_train_step(
                model, tokenizer, optimizer, 2, 1.0, r1_zero_reward_fn,
                prompts, responses, ["4"] * 4, 4,
                baseline="none", advantage_normalizer="none",
                importance_reweighting_method=method, old_log_probs=old_log_probs, cliprange=0.2,
                loss_normalization="sequence",
            )
            assert torch.isfinite(loss)
            assert old_log_probs.grad is None
            assert metadata["active_fraction"] == 0.5
            assert 0 <= float(metadata["clip_fraction"]) <= 1
            assert all(p.grad is None for p in model.parameters())
            rows.append({"method": method, "update": update, "loss": loss.item(), **{
                key: float(value) for key, value in metadata.items()
            }})
    return (rows,)


@app.cell
def _(mo, rows):
    mo.ui.table(rows)
    return


if __name__ == "__main__":
    app.run()
