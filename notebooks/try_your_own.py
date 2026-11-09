import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import torch

    from cs336_alignment.rewards import compute_group_normalized_rewards
    from cs336_alignment.reporting import (
        load_experiments, plot_final_rewards, plot_metric_curves, summarize_final_rewards, paired_reward_differences,
    )

    return Path, compute_group_normalized_rewards, load_experiments, mo, plot_final_rewards, plot_metric_curves, summarize_final_rewards, torch, paired_reward_differences


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — try_your_own

    ## 唯一改动：leave-one-out baseline

    以 Dr. GRPO 为基线，保留无标准差归一化、常数 loss 归一化、on-policy
    更新，只将包含自身 reward 的组均值 baseline 换成其余 $G-1$ 个样本的均值：
    $$A_i^{LOO}=r_i-\frac{\sum_{j\ne i}r_j}{G-1}
      =\frac{G}{G-1}(r_i-\bar r).$$
    这是一项有限组大小的偏差修正，并非声称发明了新算法。

    令 $s_i=\nabla\log\pi_\theta(y_i\mid x)$，条件于同一个 prompt 的
    独立 rollout 有 $E[s_i]=0$、$E[r_j s_i]=0$（$j\ne i$）。因此
    $$E[(r_i-\bar r)s_i]=(1-1/G)E[r_i s_i],\qquad
      E[A_i^{LOO}s_i]=E[r_i s_i].$$
    这个无偏结论针对未裁剪的原始梯度估计量；不意味着经过梯度裁剪或 Adam
    后的参数更新也是无偏的。

    **重要局限：** 固定 $G$ 时 LOO 只是将 Dr. GRPO 的整个梯度乘以
    $G/(G-1)$；$G=8$ 时是 8/7。它不会改变 prompt 间的相对权重或梯度方向，
    原始方差也放大 $(8/7)^2$。Adam 对统一梯度缩放近似不变，梯度裁剪也可能
    抵消缩放，所以不预期仅靠这个修正一定提高验证准确率。
    """)
    return


@app.cell
def _(compute_group_normalized_rewards, torch):
    rewards = torch.tensor([1., 0., 0., 1., 0., 0., 0., 0.])
    dr_advantages, _ = compute_group_normalized_rewards(rewards, 8, "mean", advantage_normalizer="none")
    loo_advantages, _ = compute_group_normalized_rewards(rewards, 8, "loo", advantage_normalizer="none")
    direct_loo = rewards - (rewards.sum() - rewards) / 7
    torch.testing.assert_close(loo_advantages, direct_loo)
    torch.testing.assert_close(loo_advantages, dr_advantages * (8 / 7))
    return dr_advantages, loo_advantages, rewards


@app.cell
def _(dr_advantages, loo_advantages, mo, rewards):
    mo.ui.table([
        {"rollout": i, "reward": rewards[i].item(),
         "Dr_GRPO_advantage": dr_advantages[i].item(), "LOO_advantage": loo_advantages[i].item()}
        for i in range(len(rewards))
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 对照实验方案及完成状态

    模型为本地 `models/OLMo-2-0425-1B`，数据为相同 GSM8K train/validation
    切分，prompt、温度、最大生成长度、训练步数、数据顺序、评估频率、
    optimizer、学习率和 gradient clipping 均与 Dr. GRPO 配对保持一致。
    使用 seeds **0、1、2、3**，只改 baseline 为 `loo`，不重新调学习率。
    两组都用 `advantage_normalizer="none"` 和 `loss_normalization="constant"`。

    应报告每个 seed 的 validation reward、跨 seed 均值和标准差、配对差值，
    并绘制 validation reward 随 optimizer step 和 wall time 的曲线；
    同时检查 entropy、response length、gradient norm 和 truncation rate。
    不能把单次最高分作为优于基线的证据。

    下方读取真实运行状态；只有完整四种子最终评估才能作为完整实验结论。
    `failed`、`diverged`、`incomplete` 与成功完成区分，不把缺失结果当作零分。
    上面的张量验证仅证明 baseline 实现与推导一致，不能据此声称超过 Dr. GRPO。
    """)
    return


@app.cell
def _(Path, load_experiments):
    own_result_root = Path(__file__).resolve().parents[1] / "results"
    own_runs, own_observations = load_experiments({
        "Dr. GRPO": own_result_root / "grpo_dr_grpo",
        "LOO": own_result_root / "grpo_loo",
    })
    return own_observations, own_runs


@app.cell
def _(mo, own_runs, summarize_final_rewards):
    own_summary, own_commentary = summarize_final_rewards(
        own_runs, expected_seeds=4, expected_groups=["Dr. GRPO", "LOO"],
    )
    mo.vstack([mo.md(own_commentary), mo.ui.table(own_summary), mo.ui.table([
        {"method": run["label"], "seed": run["seed"], "status": run["status"],
         "final_reward": run["final_reward"], "latest_reward": run["latest_reward"],
         "latest_val_step": run["latest_val_step"]}
        for run in own_runs
    ])])
    return


@app.cell
def _(own_runs, plot_final_rewards):
    plot_final_rewards(own_runs, expected_seeds=4, expected_groups=["Dr. GRPO", "LOO"])
    return


@app.cell
def _(mo, own_observations, plot_metric_curves):
    mo.vstack([
        plot_metric_curves(own_observations, metric="reward", split="val", x=axis)
        for axis in ("step", "wall_seconds")
    ])
    return


@app.cell
def _(mo, own_observations, plot_metric_curves):
    mo.vstack([
        plot_metric_curves(own_observations, metric=metric, split=split, x="step")
        for metric, split in (("response_length", "val"), ("truncation_rate", "val"),
                              ("token_entropy", "train"), ("grad_norm", "train"),
                              ("active_fraction", "train"))
    ])
    return


@app.cell
def _(mo, own_runs, paired_reward_differences):
    paired_rows, paired_summary = paired_reward_differences(own_runs, "Dr. GRPO", "LOO")
    mo.vstack([mo.md("配对差值 = LOO − Dr. GRPO；仅统计两者都完成的相同 seed。"),
               mo.ui.table(paired_rows), mo.ui.table([paired_summary])])
    return


if __name__ == "__main__":
    app.run()
