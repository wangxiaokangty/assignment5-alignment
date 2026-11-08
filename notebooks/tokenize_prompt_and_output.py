import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from transformers import AutoTokenizer
    from cs336_alignment.policy import tokenize_prompt_and_output

    return AutoTokenizer, Path, mo, tokenize_prompt_and_output


@app.cell
def _(mo):
    mo.md(r"""
    # Problem 3 — tokenize_prompt_and_output

    分别以 `add_special_tokens=False` 编码 prompt 和 response，再直接拼接，
    不插入 BOS、EOS 或分隔符。整个 batch 右侧补齐，然后生成
    `input_ids = tokens[:, :-1]`、`labels = tokens[:, 1:]`。

    mask 按 **label 的来源位置** 标记，response 为 True，prompt 和 padding 为 False；
    不能用 token ID 判断 padding，因为真实 response 也可能含有相同 ID。
    首个 response token 由 prompt 的最后一个 token 预测，不应漏掉。
    空 response 的 mask 全为 False；空 prompt 的首个 token 没有前文，移位后不参与预测。

    下面直接使用 `models/OLMo-2-0425-1B` 的 tokenizer，展示不同长度与空 response。
    """)
    return


@app.cell
def _(AutoTokenizer, Path, tokenize_prompt_and_output):
    root = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained(root / "models/OLMo-2-0425-1B", local_files_only=True)
    prompts = ["Question: What is 2 + 2?\nAnswer:", "Hello", "Empty response:"]
    responses = [" 4.", " world!", ""]
    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    return batch, prompts, responses, tokenizer


@app.cell
def _(batch, mo, tokenizer):
    mo.ui.table([
        {
            "example": i,
            "position": j,
            "input_token": tokenizer.convert_ids_to_tokens(int(batch["input_ids"][i, j])),
            "label_token": tokenizer.convert_ids_to_tokens(int(batch["labels"][i, j])),
            "response_mask": bool(batch["response_mask"][i, j]),
        }
        for i in range(batch["input_ids"].shape[0])
        for j in range(batch["input_ids"].shape[1])
    ], page_size=30)
    return


@app.cell
def _(batch, mo, responses, tokenizer):
    expected_counts = [len(tokenizer.encode(r, add_special_tokens=False)) for r in responses]
    actual_counts = batch["response_mask"].sum(dim=1).tolist()
    assert actual_counts == expected_counts
    assert not batch["response_mask"][2].any()
    mo.md(f"三个张量 shape：`{tuple(batch['input_ids'].shape)}`；response token 数量：`{actual_counts}`。对齐检查通过。")
    return


if __name__ == "__main__":
    app.run()
