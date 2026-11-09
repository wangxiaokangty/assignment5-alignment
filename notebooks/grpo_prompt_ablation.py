import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from cs336_alignment.reporting import (
        load_experiments,
        plot_final_rewards,
        plot_metric_curves,
        summarize_final_rewards,
    )

    return Path, load_experiments, mo, plot_final_rewards, plot_metric_curves, summarize_final_rewards


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — grpo_prompt_ablation

    比较 zero-shot `r1_zero`、`question_only` 和 `r1_zero_three_shot`，
    每种 prompt 使用 seeds 0、1、2、3。模型、数据划分、学习率、训练步数、
    rollout/group size、温度、生成长度、评估频率和 GRPO 配置应保持一致。
    Prompt 及其规定的停止规则和 grader 随 prompt 改变。

    下列图表直接读取本地真实训练日志；未完成、失败或缺少最终评估的 run
    不能纳入最终 reward 统计。中途曲线只描述当前进度，不当作最终性能。
    每个点表示一个 seed；跨 seed 的样本标准差衡量训练随机性的敏感程度，
    四个 seeds 的最高均值不自动构成显著优越的证据。

    `question_only` 的答案解析比 XML 格式要求宽松，因此 reward 差异可能同时
    来自推理质量和输出格式。需一起检查 answer reward、format reward、
    response length、truncation rate 和 entropy。训练中全零优势 batch 会跳过
    更新，此时记录的 entropy=0 是实现约定，不代表模型确定性地输出。
    """)
    return


@app.cell
def _(Path, load_experiments):
    result_root = Path(__file__).resolve().parents[1] / "results"
    prompt_runs, prompt_observations = load_experiments({
        "r1_zero": result_root / "grpo_standard",
        "question_only": result_root / "grpo_prompt_ablation/question_only",
        "r1_zero_three_shot": result_root / "grpo_prompt_ablation/r1_zero_three_shot",
    })
    return prompt_observations, prompt_runs


@app.cell
def _(mo, prompt_runs, summarize_final_rewards):
    prompt_summary, prompt_commentary = summarize_final_rewards(prompt_runs, expected_seeds=4, expected_groups=["r1_zero", "question_only", "r1_zero_three_shot"])
    mo.vstack([mo.md(prompt_commentary), mo.ui.table(prompt_summary)])
    return


@app.cell
def _(plot_final_rewards, prompt_runs):
    plot_final_rewards(prompt_runs, expected_seeds=4, expected_groups=["r1_zero", "question_only", "r1_zero_three_shot"])
    return


@app.cell
def _(plot_metric_curves, prompt_observations):
    plot_metric_curves(prompt_observations, metric="reward", split="val", x="step")
    return


@app.cell
def _(plot_metric_curves, prompt_observations):
    plot_metric_curves(prompt_observations, metric="reward", split="val", x="wall_seconds")
    return


@app.cell
def _(mo, plot_metric_curves, prompt_observations):
    mo.vstack([
        plot_metric_curves(prompt_observations, metric=metric, split="val", x="step")
        for metric in ("answer_reward", "format_reward", "response_length", "truncation_rate")
    ])
    return


@app.cell
def _(mo, plot_metric_curves, prompt_observations):
    mo.vstack([
        plot_metric_curves(prompt_observations, metric=metric, split="train", x="step")
        for metric in ("token_entropy", "active_fraction", "grad_norm")
    ])
    return


if __name__ == "__main__":
    app.run()
