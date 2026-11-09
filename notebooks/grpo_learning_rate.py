import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    from cs336_alignment.reporting import load_experiments, plot_final_rewards, summarize_final_rewards
    return Path, load_experiments, mo, plot_final_rewards, summarize_final_rewards


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — grpo_learning_rate

    扫描学习率 $3\times10^{-6},10^{-5},3\times10^{-5}$，每个配置计划运行 4 个种子。
    较小与较大的学习率覆盖默认值两侧；其余训练与验证配置保持一致。
    图中每个点是一条确认完成的运行，黑线是均值和最小/最大范围，不是置信区间。
    未完成和失败运行单独显示，绝不使用中途验证奖励替代最终奖励。

    四种子的设计与标准 GRPO 的多种子基线保持一致，便于观测运行间方差；
    仅四个种子不能可靠识别微小收益。验证集用于选学习率后的结果也不是独立泛化评估。
    重新运行 notebook 会读取当前磁盘上的真实结果；本报告不启动或重启训练。
    """)
    return


@app.cell
def _(Path, load_experiments, summarize_final_rewards):
    root = Path(__file__).resolve().parents[1]
    runs, observations = load_experiments({
        "default": root / "results/grpo_standard",
        "sweep": root / "results/grpo_learning_rate",
    })
    summary, commentary = summarize_final_rewards(
        runs, group_by="learning_rate", expected_seeds=4, expected_groups=[3e-6, 1e-5, 3e-5]
    )
    return commentary, runs, summary


@app.cell
def _(plot_final_rewards, runs):
    plot_final_rewards(runs, group_by="learning_rate", expected_seeds=4, expected_groups=[3e-6, 1e-5, 3e-5])
    return


@app.cell
def _(mo, summary):
    mo.ui.table(summary)
    return


@app.cell
def _(mo, runs):
    mo.ui.table([{
        key: run.get(key) for key in (
            "learning_rate", "seed", "status", "steps", "final_reward", "latest_val_step",
            "latest_reward", "zero_update_fraction", "failure",
        )
    } for run in runs])
    return


@app.cell
def _(commentary, mo):
    mo.md(commentary)
    return


if __name__ == "__main__":
    app.run()
