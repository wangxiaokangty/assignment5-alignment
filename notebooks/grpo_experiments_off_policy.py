import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    from cs336_alignment.reporting import load_experiments, summarize_final_rewards, plot_final_rewards, plot_metric_curves, summarize_runtime
    return Path, mo, load_experiments, summarize_final_rewards, plot_final_rewards, plot_metric_curves, summarize_runtime


@app.cell
def _(mo):
    mo.md(r"""# Problem — grpo_experiments_off_policy

    保持标准实验的模型、r1_zero、学习率 $10^{-5}$、256 rollouts、G=8、512 token 上限、
    200 rollout steps、1024 验证问题和 4 seeds。
    Off-policy 将 train batch 改为8、gradient accumulation 改为1：
    每个 rollout batch 顺序执行32次更新，每次保留完整的8响应组。
    所有旧 log-prob 在第一个更新前由固定旧策略计算，并停止梯度。
    GRPO cliprange=0.2，GSPO cliprange=0.0003。

    同时按 rollout step 和真实 wall time 比较，避免把32次更新误当成同等训练计算量。
    日志中的 `update_seconds` 包含旧概率评分、奖励与反传，
    `optimizer_calls` 包含跳过的零优势批次，`active_optimizer_calls` 才是实际更新次数。
    为统一计时，naive 也执行旧概率评分，但不使用重要性比率。
    `clip_fraction` 是 active response token 上实际裁剪分支的比例，再对32个调用取均值；
    全零优势调用为0。GSPO 的整条序列共享裁剪决策，按 token 数加权后记录，
    因而它与 GRPO 的裁剪粒度不同，不能仅凭较低的比例宣称更稳定。

    未完成 run 不进入最终统计。观察验证奖励、种子方差、熵、梯度和耗时后才判断效果，
    同一学习率也未必是所有 off-policy 方法各自的最优值。
    """)
    return


@app.cell
def _(load_experiments):
    directories = {'standard': 'results/grpo_standard', 'naive': 'results/grpo_offpolicy_naive', 'noclip': 'results/grpo_offpolicy_noclip', 'GRPO_clip': 'results/grpo_offpolicy_clip', 'GSPO': 'results/grpo_offpolicy_gspo'}
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
    output = Path("results/grpo_experiments_off_policy/plots")
    output.mkdir(parents=True, exist_ok=True)
    final_plot = plot_final_rewards(runs, expected_groups=list(directories))
    final_plot.savefig(output / "final_reward.svg")
    final_plot
    return (output,)


@app.cell
def _(observations, plot_metric_curves, output, mo):
    plots = []
    for split, metric in [('val', 'reward'), ('val', 'response_length'), ('train', 'reward'), ('train', 'grad_norm'), ('train', 'token_entropy'), ('train', 'clip_fraction'), ('train', 'active_fraction'), ('train', 'update_seconds')]:
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


@app.cell
def _(runs, observations, summarize_runtime, mo):
    runtime = summarize_runtime(runs, observations)
    mo.vstack([mo.md("完整运行的实测耗时；尚未完成的 run 不用于最终速度比较。"), mo.ui.table(runtime)])
    return


if __name__ == "__main__":
    app.run()
