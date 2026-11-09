import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    from cs336_alignment.reporting import load_experiments, summarize_final_rewards, plot_final_rewards, plot_metric_curves
    return Path, mo, load_experiments, summarize_final_rewards, plot_final_rewards, plot_metric_curves


@app.cell
def _(mo):
    mo.md(r"""# Problem — grpo_experiments_variants_on_policy

    固定 OLMo-2-0425-1B、r1_zero、学习率 $10^{-5}$、200 rollout steps、256 responses/batch、
    G=8、512 token 上限、4 seeds 和 1024 验证问题。
    GRPO_constant 仅将 sequence 平均改为 $Z=256\times512$；Dr_GRPO 再去掉 std；
    RFT 进一步去掉 baseline；MaxRL 从 GRPO_constant 改为 mean normalizer。
    所有方法均只对非零优势回答反传，保持原始 batch 分母。

    下面显示实际生成的曲线及每个 seed 最终结果；未完成的实验只显示中途曲线，不进入最终统计。
    固定学习率控制变量，但不代表各方法都已分别调到最优；表现较差不能证明方法本身更差。
    只有四个 seeds，均值/样本标准差和 min/max 是描述性证据。
    `token_entropy` 统计 active responses，零更新时的占位值已从熵曲线中剔除。
    """)
    return


@app.cell
def _(load_experiments):
    directories = {'standard': 'results/grpo_standard', 'GRPO_constant': 'results/grpo_grpo_constant', 'Dr_GRPO': 'results/grpo_dr_grpo', 'RFT': 'results/grpo_rft', 'MaxRL': 'results/grpo_maxrl'}
    runs, observations = load_experiments(directories)
    return directories, runs, observations


@app.cell
def _(directories, runs, summarize_final_rewards, mo):
    summary, commentary = summarize_final_rewards(runs, expected_groups=list(directories))
    mo.vstack([mo.md(commentary), mo.ui.table(summary), mo.ui.table([
        {key: row.get(key) for key in ("label", "seed", "status", "latest_val_step", "latest_reward", "final_reward", "zero_update_fraction", "failure")}
        for row in runs
    ])])
    return


@app.cell
def _(Path, directories, runs, plot_final_rewards):
    output = Path("results/grpo_experiments_variants_on_policy/plots")
    output.mkdir(parents=True, exist_ok=True)
    final_plot = plot_final_rewards(runs, expected_groups=list(directories))
    final_plot.savefig(output / "final_reward.svg")
    final_plot
    return (output,)


@app.cell
def _(observations, plot_metric_curves, output, mo):
    plots = []
    for split, metric in [('val', 'reward'), ('val', 'format_reward'), ('val', 'response_length'), ('train', 'reward'), ('train', 'grad_norm'), ('train', 'token_entropy'), ('train', 'active_fraction')]:
        figure = plot_metric_curves(observations, metric, split)
        figure.savefig(output / f"{split}_{metric}.svg")
        plots.append(figure)
    mo.vstack(plots)
    return


@app.cell
def _(observations, plot_metric_curves, output):
    wall_plot = plot_metric_curves(observations, "reward", "val", "wall_seconds")
    wall_plot.savefig(output / "reward_wall_seconds.svg")
    wall_plot
    return


if __name__ == "__main__":
    app.run()
