import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cs336_alignment.policy import get_response_log_probs, tokenize_prompt_and_output

    return AutoModelForCausalLM, AutoTokenizer, Path, get_response_log_probs, mo, tokenize_prompt_and_output, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Problem 4 — get_response_log_probs

    对模型 logits 在词表维度做 `log_softmax`，再用 `gather` 取出 label 的
    条件 log-probability。labels 已由第 3 题移位，这里不能再次移位。
    可选的逐位置熵为 $H(p)=-\sum_v p_v\log p_v$，单位是 nat。
    半精度 logits 转为 float32 后计算，避免低精度归一化和求和带来的误差。

    函数保留梯度和模型模式，返回所有位置的结果；调用方用 `response_mask`
    排除 prompt 和 padding。`eval()` 仅关闭 dropout，不会关闭梯度。

    下面使用本地 `models/OLMo-2-0425-1B`，展示真实 response token 的分数，
    并对 response 的平均负 log-probability 做一次反向传播（不更新参数）。
    默认使用可见的第一张 GPU，可在启动前用 `CUDA_VISIBLE_DEVICES` 选择。
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, Path, torch):
    model_path = Path(__file__).resolve().parents[1] / "models/OLMo-2-0425-1B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    return device, model, tokenizer


@app.cell
def _(device, tokenize_prompt_and_output, tokenizer):
    prompts = ["Question: What is 2 + 2?\nAnswer:", "The capital of France is"]
    responses = [" 4.", " Paris."]
    batch = {
        key: value.to(device)
        for key, value in tokenize_prompt_and_output(prompts, responses, tokenizer).items()
    }
    return (batch,)


@app.cell
def _(batch, get_response_log_probs, model, torch):
    model.zero_grad(set_to_none=True)
    scores = get_response_log_probs(model, batch["input_ids"], batch["labels"], True)
    assert scores["log_probs"].shape == batch["labels"].shape
    assert scores["token_entropy"].shape == batch["labels"].shape
    assert torch.isfinite(scores["log_probs"]).all()
    assert (scores["token_entropy"] >= 0).all()
    loss = -scores["log_probs"][batch["response_mask"]].mean()
    loss.backward()
    grad_norm = model.get_input_embeddings().weight.grad.float().norm().item()
    assert 0 < grad_norm < float("inf")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        without_entropy = get_response_log_probs(model, batch["input_ids"], batch["labels"])
    assert set(without_entropy) == {"log_probs"}
    torch.testing.assert_close(without_entropy["log_probs"], scores["log_probs"])
    return grad_norm, loss, scores


@app.cell
def _(batch, mo, scores, tokenizer):
    mo.ui.table([
        {
            "example": i,
            "position": j,
            "response_token": tokenizer.convert_ids_to_tokens(int(batch["labels"][i, j])),
            "log_probability": scores["log_probs"][i, j].item(),
            "entropy_nats": scores["token_entropy"][i, j].item(),
        }
        for i, j in batch["response_mask"].nonzero().tolist()
    ])
    return


@app.cell
def _(device, grad_norm, loss, mo, scores):
    mo.md(f"""
    实际设备：`{device}`；输出 shape：`{tuple(scores['log_probs'].shape)}`。

    Response 平均负 log-probability：**{loss.item():.4f}**；
    embedding 梯度范数：**{grad_norm:.4f}**。
    shape、有限值、非负熵、默认不返回 entropy，以及梯度回传检查均已通过。
    """)
    return


if __name__ == "__main__":
    app.run()
