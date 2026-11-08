import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from cs336_alignment.training import load_rollout_examples, load_training_results, plot_metric
    from cs336_alignment.reporting import load_experiments, summarize_final_rewards
    return load_rollout_examples, load_training_results, plot_metric, mo, load_experiments, summarize_final_rewards


@app.cell
def _(mo):
    mo.md(r"""
    # grpo_experiments_standard_on_policy

    本地训练入口：`uv run python scripts/train_grpo.py`。默认使用 GPU 0 训练、GPU 1
    生成，4 个 seed 顺序运行；通过 `--policy-device` 和 `--inference-gpu` 指定显卡。
    vLLM 的 GPU 参数是物理 GPU 编号，训练设备编号相对于当前 `CUDA_VISIBLE_DEVICES`。
    默认采用题目要求的 200 次更新、6400 条训练问题、1024 条验证问题、每批 256 回答、
    每题 8 回答、32 次梯度累积、512 token 上限、学习率 $10^{-5}$。

    每次更新后同步权重，验证 step 0 是原始模型，step 200 是最后更新后的模型。
    每 10 步验证，每 40 步保存完整训练/验证 rollouts；首批与末次验证也保留。
    `config.json` 记录配置与数据指纹，`metrics.jsonl` 记录实际指标和累计 wall time，
    `rollouts.jsonl` 记录回答，`completed.json` 仅在完整运行成功后写入。
    每 10 步保存包含模型、优化器和随机状态的滚动 `resume.pt`，中断后使用同一配置
    并增加 `--resume` 恢复；最多重做最近一个验证间隔。
    下方图表从实际结果加载，各 seed 的曲线和均值、min/max 区间同时显示。
    零优势回答在反传前剪除；熵仅统计 active responses，跳过更新时的0是占位，
    并非策略熵真的为0。`active_fraction` 用于辨别哪些 batch 没有学习信号。
    **短 smoke 只用于验证运行链路，不作为完整实验或正确率达标的证据。**
    """)
    return


@app.cell
def _(load_training_results, mo):
    observations, run_status = load_training_results()
    mo.ui.table(run_status) if run_status else mo.md("尚无标准规模实验结果；不能判断是否达到平均 25% 的验证正确率。")
    return observations, run_status


@app.cell
def _(plot_metric, mo, observations):
    mo.vstack([plot_metric(observations, split, metric, f"results/grpo_standard/plots/{split}_{metric}.svg") for split, metric in [
        ("train", "loss"), ("train", "grad_norm"), ("train", "token_entropy"),
        ("train", "reward"), ("train", "format_reward"), ("val", "reward"),
        ("val", "format_reward"), ("val", "response_length"),
    ]])
    return


@app.cell
def _(mo, observations):
    mo.ui.table([row for row in observations if row["split"] == "val" and row["step"] in (0, 50, 200)])
    return


@app.cell
def _(load_rollout_examples, mo):
    mo.vstack([
        mo.md("## 同一验证问题的最早与最近一次回答（seed 0）\n\n未完成训练时，最近一次不代表最终结果。"),
        mo.ui.table(load_rollout_examples()),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 先导阶段人工检查（真实 step 40）

    seed 0 的验证正确率由 step 0 的 1/1024 升到 step 40 的 281/1024（27.44%），
    seed 1 在 step 20 达到293/1024（28.61%）。这些是训练中途证据，最终结论见上表。
    检查 seed 0 同一验证问题的前后回答：索引9正确计算加班工资 400+60=460，
    索引11正确汇总糕点收入 204+160+330=694，索引22正确计算 DVD 数 3+4=7。
    仍有错误样本：固定索引0/1/2/10/100在step40并未全部解决，存在添造成本、
    重复计数和漏标签；索引3最终数字正确但叙述单位错误。
    因此奖励提升是实际可观察的，不能据最终答案奖励推断所有推理步骤都正确。
    回答表保留最早几个固定问题的前后回答，不只挑选成功样本。
    """)
    return


@app.cell
def _(load_experiments, summarize_final_rewards, mo):
    standard_runs, _ = load_experiments({"standard": "results/grpo_standard"})
    standard_summary, standard_commentary = summarize_final_rewards(standard_runs, expected_groups=["standard"])
    mo.vstack([mo.md(standard_commentary), mo.ui.table(standard_summary)])
    return


if __name__ == "__main__":
    app.run()
