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
    # Problem — compute_group_normalized_rewards_grpo

    将同一问题的回复按连续的 `group_size` 个元素分组，计算
    $A_i=(r_i-\bar r)/(s+\epsilon)$，其中 $s$ 使用样本标准差（分母为 $G-1$）。
    输出恢复为一维 rollout 顺序，保留输入的设备和 dtype。

    以下包含奖励有差异、全部成功和全部失败的组。
    后两种组没有可区分的相对表现，因此 advantage 均为零；epsilon 防止除零。
    单元素组也返回零 advantage。这是纯张量运算，无需加载模型。
    """)
    return


@app.cell
def _(compute_group_normalized_rewards, torch):
    raw_rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    advantages, metadata = compute_group_normalized_rewards(raw_rewards, group_size=2)
    expected_magnitude = 0.5 / (2**-0.5 + 1e-6)
    torch.testing.assert_close(
        advantages,
        torch.tensor([expected_magnitude, -expected_magnitude, 0.0, 0.0, 0.0, 0.0]),
    )
    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(advantages.reshape(-1, 2).mean(dim=1), torch.zeros(3))
    singleton_advantages, _ = compute_group_normalized_rewards(torch.tensor([1.0]), group_size=1)
    torch.testing.assert_close(singleton_advantages, torch.zeros(1))
    return advantages, metadata, raw_rewards


@app.cell
def _(advantages, mo, raw_rewards):
    mo.ui.table([
        {"group": index // 2, "reward": reward, "advantage": advantage}
        for index, (reward, advantage) in enumerate(zip(raw_rewards.tolist(), advantages.tolist()))
    ])
    return


@app.cell
def _(metadata, mo):
    mo.md(f"实际统计：`{metadata}`。数值、组内零均值、常量组与单元素组检查通过。")
    return


if __name__ == "__main__":
    app.run()
