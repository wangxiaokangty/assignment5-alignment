import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch

    from cs336_alignment.losses import compute_policy_gradient_loss

    return compute_policy_gradient_loss, mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — compute_policy_gradient_loss_on_policy

    On-policy 时每个 token 的损失为 $\ell_{it}=-A_i\log\pi_\theta(y_{it})$。
    PyTorch 使用梯度下降，因此需要负号；正 advantage 提高采样 token 的概率，
    负 advantage 降低该概率。优势值是固定的 rollout 目标，不参与求导。
    核心实现返回未聚合的损失，response mask 和平均操作由后续题处理。

    以下用两个实际张量样本验证广播和梯度，正负 advantage 分别为 2 和 −1。
    """)
    return


@app.cell
def _(compute_policy_gradient_loss, torch):
    advantages = torch.tensor([2.0, -1.0])
    log_probs = torch.tensor([[-0.2, -0.5, -1.0], [-0.3, -0.7, -1.2]], requires_grad=True)
    per_token_loss, metadata = compute_policy_gradient_loss(advantages, log_probs)
    column_loss, _ = compute_policy_gradient_loss(advantages[:, None], log_probs)
    torch.testing.assert_close(per_token_loss, column_loss)
    per_token_loss.sum().backward()
    torch.testing.assert_close(log_probs.grad, -advantages[:, None].expand_as(log_probs))
    return advantages, log_probs, per_token_loss


@app.cell
def _(advantages, log_probs, mo, per_token_loss):
    mo.ui.table([
        {
            "sequence": i,
            "advantage": advantages[i].item(),
            "log_probs": log_probs[i].detach().tolist(),
            "per_token_loss": per_token_loss[i].detach().tolist(),
            "gradient_wrt_log_probs": log_probs.grad[i].tolist(),
        }
        for i in range(len(advantages))
    ])
    return


if __name__ == "__main__":
    app.run()
