import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json

    import marimo as mo

    from cs336_alignment.prompting import (
        ROOT,
        PROMPTS,
        audit_examples,
        load_results,
        run_evaluation,
    )

    return PROMPTS, ROOT, audit_examples, json, load_results, mo, run_evaluation


@app.cell
def _(mo):
    mo.md(r"""
    # Problem 1 — prompting_baselines

    使用本地 `models/OLMo-2-0425-1B`，比较 GSM8K test 上的三种 prompt。
    每题采样一次：temperature=1.0，top-p=1.0，max_tokens=512，seed=0。
    R1 两种 prompt 遇到 `</answer>` 停止并保留该标签；question_only 不设此停止串。
    使用原作业评分器，ground truth 取 `####` 后的答案。

    完整重跑（在项目根目录）：
    ```bash
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv/bin/marimo edit notebooks/prompting_baselines.py
    ```
    默认读取已保存的真实结果。点击下方按钮可在本地 GPU 上重新评测并覆盖结果；
    人工检查记录及文字分析对应已保存的原始实验，重跑后需重新核对。
    """)
    return


@app.cell
def _(mo):
    rerun = mo.ui.run_button(label="重新运行完整 GSM8K 评测（本地 GPU）")
    rerun
    return (rerun,)


@app.cell
def _(ROOT, rerun, run_evaluation):
    if rerun.value:
        run_evaluation(
            model=ROOT / "models/OLMo-2-0425-1B",
            dataset=ROOT / "data/gsm8k/test.jsonl",
            output=ROOT / "results/prompting_baselines",
        )
    evaluation_finished = True
    return (evaluation_finished,)


@app.cell
def _(evaluation_finished, load_results, mo):
    summaries, records = load_results()
    mo.ui.table([{"prompt": name, **metrics} for name, metrics in summaries.items()])
    return records, summaries


@app.cell
def _(mo):
    mo.md(r"""
    ## (a) 指标与人工检查

    类别 1：格式与答案均正确；类别 2：格式正确、评分器判答案错误；类别 3：格式不合规。
    这是原评分器的结果，未用人工判断修改分数。

    | Prompt | 类别 1 | 类别 2 | 类别 3 | 正确率 | 格式通过率 | 达到 512 token 上限 |
    |---|---:|---:|---:|---:|---:|---:|
    | question_only | 4 | 311 | 1004 | 0.30% | 23.88% | 429 |
    | r1_zero | 2 | 639 | 678 | 0.15% | 48.60% | 28 |
    | r1_zero_three_shot | 249 | 990 | 80 | 18.88% | 93.93% | 5 |

    按测试集顺序，对每种 prompt 的类别 2、3 各检查前十条，共 60 条。
    “语义正确但漏判”要求响应已经给出明确正确结论，不能只是在错误推导中碰巧出现正确数字。

    | Prompt | 类别 2 中语义正确 | 类别 3 中语义正确 |
    |---|---:|---:|
    | question_only | 0/10 | 0/10 |
    | r1_zero | 1/10 | 1/10 |
    | r1_zero_three_shot | 0/10 | 3/10 |

    具体例子（编号从 0 开始）：r1_zero 第 1 题回答
    `That makes a total of 3 bolts of fiber.`，数学答案正确，但评分器比较整个句子，判为类别 2。
    第 6 题正确得到 `20 + 80 + 160 = 260 sheep`，却缺少要求的标签，落入类别 3。
    三例提示第 104 题正确答 500，但把 `</think>` 写成 `</thinking>`；
    第 156 题推导得到 1400，但答案标签未完成；第 166 题正确答 5，却缺少 `</answer>`。
    它们都是格式或解析造成的漏判。

    三例提示第 118 题虽然中间出现 `20 / 5 = 4`，后续却否定计算并陷入重复，因此没有把它算作
    “只是解析失败”。这些前十条样本便于复核，但不是随机抽样，不能直接外推全测试集的漏判比例。

    ## (b) Prompt 对行为的影响

    **question_only** 很难让这个 base model 稳定进入解题模式。第 1 题生成 `[EX Q]` / `[EX A]`
    并继续编造其他题，第 3 题续写教学网页，第 11、12 题直接输出空文本。
    因此仅给出问题和 boxed 指令仍不足以保证它回答当前问题。

    **r1_zero** 明显增加了与当前问题相关的推理和标签输出，但未必生成可提取的简短答案。
    第 1 题展示了“答案正确但写成完整句子”的情况；第 22 题虽用了正确标签，却把
    `3 + 2×2` 算成了 9。格式通过率提升不代表数学正确率同样提升。

    **三例提示** 同时示范了解题过程、标签结构和短答案，使模型更经常按同一模式作答；
    正确率达到 18.88%，格式通过率达到 93.93%。但仍有大量实质性错误，例如第 10 题
    正确求出第二、三个月的 180 和 126，却漏掉第一个月的 60，答成 306；
    第 11 题把 `204 + 160 + 330` 错加为 684。
    这些例子说明 few-shot 改善任务与格式遵循，但没有消除推理和算术错误。

    以上是本地 A100、bfloat16、vLLM 0.19.1 的一次采样结果，非多 seed 平均。
    使用仓库原始 prompt 文本（包括 question_only 中的反斜线），不应用 chat template。
    仅给引擎设置全局 seed=0，不给所有请求重复设置同一 seed。
    原始响应、prompt、评分和停止原因保存在 JSONL；环境和输入摘要见 metadata.json。
    下方可交互查看全部题目，以及 60 条逐条人工判断。
    """)
    return


@app.cell
def _(ROOT, json, mo):
    audit = json.loads((ROOT / "results/prompting_baselines/audit.json").read_text())
    mo.ui.table(audit, page_size=10)
    return


@app.cell
def _(PROMPTS, mo):
    selected_prompt = mo.ui.dropdown(
        options=list(PROMPTS), value="question_only", label="Prompt"
    )
    selected_prompt
    return (selected_prompt,)


@app.cell
def _(audit_examples, mo, records, selected_prompt):
    mo.ui.table(audit_examples(records[selected_prompt.value]), page_size=10)
    return


@app.cell
def _(mo, records, selected_prompt):
    example_index = mo.ui.slider(
        0,
        len(records[selected_prompt.value]) - 1,
        value=0,
        label="测试题编号",
        show_value=True,
    )
    example_index
    return (example_index,)


@app.cell
def _(example_index, mo, records, selected_prompt):
    example = records[selected_prompt.value][example_index.value]
    mo.vstack(
        [
            mo.md(
                f"**类别 {example['category']} · 标准答案 {example['ground_truth']} · 结束原因 {example['finish_reason']}**"
            ),
            mo.md("### 实际 prompt"),
            mo.plain_text(example["prompt"]),
            mo.md("### 模型原始响应"),
            mo.plain_text(example["response"]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
