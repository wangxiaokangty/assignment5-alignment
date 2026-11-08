import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch

    from cs336_alignment.rewards import compute_group_normalized_rewards

    return compute_group_normalized_rewards, mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — compute_group_normalized_rewards_drgrpo

    Dr. GRPO 的 advantage 保留组内均值 baseline，移除标准差归一化：
    $A_i=r_i-\bar r$。这样不再因问题的组内奖励标准差不同而重新缩放更新。
    设置 `baseline="none"` 且 `advantage_normalizer="none"` 则直接使用原始奖励。

    下面用成功率分别为 1/2、1/4、1 的三组回复，比较 GRPO、Dr. GRPO 与原始奖励。
    """)
    return


@app.cell
def _(compute_group_normalized_rewards, torch):
    rewards = torch.tensor([1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.])
    grpo, _ = compute_group_normalized_rewards(rewards, group_size=4)
    drgrpo, _ = compute_group_normalized_rewards(
        rewards, group_size=4, advantage_normalizer="none"
    )
    no_baseline, _ = compute_group_normalized_rewards(
        rewards, group_size=4, baseline="none", advantage_normalizer="none"
    )
    torch.testing.assert_close(
        drgrpo, torch.tensor([.5, .5, -.5, -.5, .75, -.25, -.25, -.25, 0., 0., 0., 0.])
    )
    torch.testing.assert_close(no_baseline, rewards)
    torch.testing.assert_close(drgrpo.reshape(-1, 4).mean(dim=1), torch.zeros(3))
    return drgrpo, grpo, no_baseline, rewards


@app.cell
def _(drgrpo, grpo, mo, no_baseline, rewards):
    mo.ui.table([
        {
            "group": index // 4,
            "reward": rewards[index].item(),
            "GRPO": grpo[index].item(),
            "Dr. GRPO": drgrpo[index].item(),
            "no baseline": no_baseline[index].item(),
        }
        for index in range(rewards.numel())
    ])
    return


@app.cell
def _(mo):
    mo.md("实际计算检查通过：Dr. GRPO 组内均值为零；关闭 baseline 和归一化后输出等于原始奖励。")
    return


if __name__ == "__main__":
    app.run()
