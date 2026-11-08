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
    # Problem — compute_group_normalized_rewards_maxrl

    MaxRL 使用组内平均奖励作为归一化因子：
    $A_i=(r_i-\bar r)/(\bar r+\epsilon)$。
    对二元奖励，失败样本的 advantage 接近 -1；成功样本的 advantage
    随组内成功率下降而增大。因此低成功率问题中的成功回复获得更大权重。

    下面计算成功率为 1/2、1/4、0 的三组回复。
    全部失败时分子为零，epsilon 使输出有限且为零。
    """)
    return


@app.cell
def _(compute_group_normalized_rewards, torch):
    rewards = torch.tensor([1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
    advantages, metadata = compute_group_normalized_rewards(
        rewards, group_size=4, advantage_normalizer="mean"
    )
    expected = torch.tensor([
        .5 / (.5 + 1e-6), .5 / (.5 + 1e-6), -.5 / (.5 + 1e-6), -.5 / (.5 + 1e-6),
        .75 / (.25 + 1e-6), -.25 / (.25 + 1e-6), -.25 / (.25 + 1e-6), -.25 / (.25 + 1e-6),
        0., 0., 0., 0.,
    ])
    torch.testing.assert_close(advantages, expected)
    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(
        advantages.reshape(-1, 4).mean(dim=1), torch.zeros(3), atol=1e-7, rtol=0
    )
    return advantages, metadata, rewards


@app.cell
def _(advantages, mo, rewards):
    mo.ui.table([
        {"group": index // 4, "reward": reward, "MaxRL advantage": advantage}
        for index, (reward, advantage) in enumerate(zip(rewards.tolist(), advantages.tolist()))
    ])
    return


@app.cell
def _(metadata, mo):
    mo.md(f"实际统计：`{metadata}`。数值、有限值、组内零均值检查通过。")
    return


if __name__ == "__main__":
    app.run()
