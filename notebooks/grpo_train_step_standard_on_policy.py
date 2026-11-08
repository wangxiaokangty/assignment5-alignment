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
    # Problem — grpo_train_step_standard_on_policy

    先对整个 rollout batch 计算组优势，再拆分 microbatch。
    每个 microbatch 的 sequence 平均损失乘以其样本数 / 总样本数，
    因而累积梯度等价于完整 batch 的平均梯度，包括不能整除的情况。
    最后只裁剪一次梯度、更新一次参数并清空梯度。

    以下使用本地 OLMo 和手工指定的正确/错误回答，执行一次真实 GPU 参数更新。
    这里验证更新流程；完整采样训练、验证曲线见实验题 notebook。
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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer, tokenizer


@app.cell
def _(grpo_train_step, model, optimizer, r1_zero_reward_fn, tokenizer, torch):
    before = next(model.parameters()).detach().float().clone()
    loss, metadata = grpo_train_step(
        model, tokenizer, optimizer, 2, 1.0, r1_zero_reward_fn,
        ["Question: What is 2 + 2?\nAnswer:"] * 4,
        ["<think>Add two and two.</think> <answer>4</answer>",
         "<think>Add two and two.</think> <answer>5</answer>",
         "<think>Two plus two is four.</think> <answer>4</answer>",
         "<think>Guess.</think> <answer>3</answer>"],
        ["4"] * 4, 4,
    )
    parameter_change = (next(model.parameters()).detach().float() - before).norm().item()
    assert torch.isfinite(loss)
    assert float(metadata["grad_norm"]) > 0
    assert all(p.grad is None for p in model.parameters())
    return loss, metadata, parameter_change


@app.cell
def _(loss, metadata, mo, parameter_change):
    mo.ui.table([{
        "loss": loss.item(),
        **{key: float(value) for key, value in metadata.items()},
        "first_parameter_change_norm": parameter_change,
    }])
    return


if __name__ == "__main__":
    app.run()
