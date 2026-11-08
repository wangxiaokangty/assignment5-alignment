import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch

    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    from cs336_alignment.rewards import compute_rollout_rewards

    return compute_rollout_rewards, mo, r1_zero_reward_fn, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — compute_rollout_rewards

    逐条调用作业提供的数学评分器，保留 rollout 顺序，返回未经归一化的
    float32 reward，以及总奖励、格式奖励、答案奖励的批次均值。
    格式正确不代表总奖励为 1：只有格式与答案均正确才能获得总奖励。

    下面直接评分两道算术题的四条示例回复，每题的 ground truth 重复两次。
    该步骤无需加载模型或占用 GPU；训练时由调用方将奖励移到训练设备。
    """)
    return


@app.cell
def _(compute_rollout_rewards, r1_zero_reward_fn, torch):
    responses = [
        "2 + 2 = 4.</think> <answer>4</answer>",
        "2 + 2 = 5.</think> <answer>5</answer>",
        "3 * 3 = 9.</think> <answer>9</answer>",
        "9",
    ]
    ground_truths = ["4", "4", "9", "9"]
    rewards, metadata = compute_rollout_rewards(r1_zero_reward_fn, responses, ground_truths)
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert metadata == {
        "mean_reward": 0.5,
        "mean_format_reward": 0.75,
        "mean_answer_reward": 0.5,
    }
    return ground_truths, metadata, responses, rewards


@app.cell
def _(ground_truths, mo, responses, rewards):
    mo.ui.table([
        {"response": response, "ground_truth": truth, "reward": reward}
        for response, truth, reward in zip(responses, ground_truths, rewards.tolist())
    ])
    return


@app.cell
def _(metadata, mo):
    mo.md(f"实际评分统计：`{metadata}`。奖励与统计检查均已通过。")
    return


if __name__ == "__main__":
    app.run()
