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
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    return AutoModelForCausalLM, AutoTokenizer, Path, grpo_train_step, mo, r1_zero_reward_fn, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — grpo_train_step_variants_on_policy

    支持 GRPO constant、Dr. GRPO、RFT、MaxRL，并在模型前向计算前移除零 advantage 回复。
    sequence 聚合仍除以原始 rollout batch 大小；constant 聚合的每个微批直接除以同一个
    $Z$ 后累加，不能再除以梯度累积次数，否则会改变梯度尺度。

    `token_entropy` 只统计保留的 active response tokens，`active_fraction` 是保留的回复比例；
    该熵不代表被剪除回复的熵。全零 advantage 批次返回零 loss、grad norm，熵记为零，
    清空梯度并跳过 optimizer.step，避免 Adam 动量或 weight decay 在无奖励信号时更新模型。

    下面使用本地 OLMo，依次进行四种真实更新，再验证全零批次不会改变参数。
    这些是更新流程演示；公平的效果比较应分别从同一初始 checkpoint 开始训练。
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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01)
    return model, optimizer, tokenizer


@app.cell
def _(grpo_train_step, model, optimizer, r1_zero_reward_fn, tokenizer, torch):
    variants = {
        "GRPO_constant": ("mean", "std"),
        "Dr_GRPO": ("mean", "none"),
        "RFT": ("none", "none"),
        "MaxRL": ("mean", "mean"),
    }
    responses = [
        "<think>Add.</think> <answer>4</answer>",
        "<think>Guess.</think> <answer>5</answer>",
        "<think>Add.</think> <answer>4</answer>",
        "<think>Guess.</think> <answer>3</answer>",
    ]
    rows = []
    for name, (baseline, normalizer) in variants.items():
        loss, metadata = grpo_train_step(
            model, tokenizer, optimizer, 2, 1.0, r1_zero_reward_fn,
            ["Question: What is 2 + 2?\nAnswer:"] * 4,
            responses, ["4"] * 4, 4,
            baseline=baseline, advantage_normalizer=normalizer,
            loss_normalization="constant", normalization_constant=4 * 32,
        )
        assert torch.isfinite(loss)
        assert float(metadata["grad_norm"]) > 0
        assert metadata["active_fraction"] == (0.5 if name == "RFT" else 1.0)
        assert all(p.grad is None for p in model.parameters())
        rows.append({"variant": name, "loss": loss.item(), **{
            key: float(value) for key, value in metadata.items()
        }})
    return (rows,)


@app.cell
def _(grpo_train_step, model, optimizer, r1_zero_reward_fn, rows, tokenizer, torch):
    # The existing momentum makes skipping the optimizer on zero signal observable.
    assert len(rows) == 4
    before = next(model.parameters()).detach().clone()
    zero_loss, zero_stats = grpo_train_step(
        model, tokenizer, optimizer, 2, 1.0, r1_zero_reward_fn,
        ["2 + 2?"] * 4, ["<think>Add.</think> <answer>4</answer>"] * 4,
        ["4"] * 4, 4,
    )
    assert zero_loss.item() == 0 and zero_stats["active_fraction"] == 0
    assert torch.equal(before, next(model.parameters()).detach())
    assert all(p.grad is None for p in model.parameters())
    return


@app.cell
def _(mo, rows):
    mo.ui.table(rows)
    return


if __name__ == "__main__":
    app.run()
