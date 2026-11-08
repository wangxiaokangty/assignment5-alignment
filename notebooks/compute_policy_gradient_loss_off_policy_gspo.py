import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch

    from cs336_alignment.losses import aggregate_loss_across_microbatch, compute_policy_gradient_loss

    return aggregate_loss_across_microbatch, compute_policy_gradient_loss, mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — compute_policy_gradient_loss_off_policy_gspo

    GSPO 先计算有效 response token 的平均 log-ratio，再取指数：
    $s_i=\exp(\sum_t m_{it}(\log\pi_\theta-\log\pi_{old})/L_i)$。
    这是 sequence ratio 的几何平均，避免直接连乘。裁剪目标仍是
    $-\min(A_i s_i,A_i\mathrm{clip}(s_i,1-\epsilon,1+\epsilon))$。

    将同一 sequence loss 广播到各 token 后，外层 sequence 平均保持该 loss
    不变；梯度里的 $1/L_i$ 已由平均 log-ratio 引入，不能再次除以长度。
    下例第一条有效 ratio 为 0.5 和 2，几何均值为 1；第二条均为 1.5，
    正优势下会被 1.2 上界裁剪。mask 排除的极端值不影响结果或梯度。
    """)
    return


@app.cell
def _(aggregate_loss_across_microbatch, compute_policy_gradient_loss, torch):
    ratios = torch.tensor([[10.0, 0.5, 2.0, 10.0], [10.0, 1.5, 1.5, 1.5]])
    mask = torch.tensor([[False, True, True, False], [False, True, True, True]])
    old_log_probs = torch.full((2, 4), -4.0, requires_grad=True)
    log_probs = (old_log_probs.detach() + ratios.log()).requires_grad_()
    advantages = torch.ones(2, requires_grad=True)
    token_losses, metadata = compute_policy_gradient_loss(advantages, log_probs, "gspo", old_log_probs, 0.2, mask)
    loss = aggregate_loss_across_microbatch(token_losses, mask)
    loss.backward()
    torch.testing.assert_close(token_losses, torch.tensor([[-1.0] * 4, [-1.2] * 4]))
    torch.testing.assert_close(loss, torch.tensor(-1.1))
    torch.testing.assert_close(log_probs.grad, torch.tensor([[0.0, -0.25, -0.25, 0.0], [0.0] * 4]))
    assert old_log_probs.grad is None and advantages.grad is None
    torch.testing.assert_close(metadata["clip_fraction"], torch.tensor(3 / 5))
    return log_probs, loss, mask, metadata, ratios, token_losses


@app.cell
def _(log_probs, loss, mask, metadata, mo, ratios, token_losses):
    mo.vstack([
        mo.md(f"实际 sequence 平均损失：**{loss.item():.3f}**；有效 token 裁剪比例：**{metadata['clip_fraction'].item():.0%}**。"),
        mo.ui.table([
            {
                "sequence": i,
                "token_ratios": ratios[i].tolist(),
                "mask": mask[i].tolist(),
                "broadcast_loss": token_losses[i].detach().tolist(),
                "gradient": log_probs.grad[i].tolist(),
            }
            for i in range(2)
        ]),
    ])
    return


if __name__ == "__main__":
    app.run()
