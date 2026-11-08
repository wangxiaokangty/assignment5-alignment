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
    # Problem — compute_policy_gradient_loss_off_policy

    Token importance ratio 为 $r_t=\exp(\log\pi_\theta-\log\pi_{old})$。
    `noclip` 的 loss 为 $-r_t A$；GRPO 使用
    $-\min(r_tA,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A)$。
    old log-probabilities 和 advantages 都是固定目标，不对它们求导。

    下面令 ratio 为 0.5、1、1.5，cliprange=0.2。
    正优势只在 ratio 太大时停止梯度，负优势只在 ratio 太小时停止梯度。
    `clip_fraction` 统计实际选择裁剪分支的位置比例，可以用 response mask
    排除 padding，而不是统计所有超出区间的 ratio。
    """)
    return


@app.cell
def _(compute_policy_gradient_loss, torch):
    ratios = torch.tensor([[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]])
    old_log_probs = torch.full((2, 3), -2.0, requires_grad=True)
    current_log_probs = (old_log_probs.detach() + ratios.log()).requires_grad_()
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    unclipped_loss, _ = compute_policy_gradient_loss(advantages, current_log_probs, "noclip", old_log_probs)
    clipped_loss, metadata = compute_policy_gradient_loss(advantages, current_log_probs, "grpo", old_log_probs, 0.2)
    unclipped_gradient = torch.autograd.grad(unclipped_loss.sum(), current_log_probs, retain_graph=True)[0]
    clipped_loss.sum().backward()
    torch.testing.assert_close(unclipped_gradient, torch.tensor([[-0.5, -1.0, -1.5], [0.5, 1.0, 1.5]]))
    torch.testing.assert_close(current_log_probs.grad, torch.tensor([[-0.5, -1.0, 0.0], [0.0, 1.0, 1.5]]))
    torch.testing.assert_close(metadata["clip_fraction"], torch.tensor(1 / 3))
    assert old_log_probs.grad is None and advantages.grad is None
    return clipped_loss, current_log_probs, metadata, ratios, unclipped_gradient, unclipped_loss


@app.cell
def _(clipped_loss, current_log_probs, metadata, mo, ratios, unclipped_gradient, unclipped_loss):
    mo.vstack([
        mo.md(f"实际裁剪比例：**{metadata['clip_fraction'].item():.1%}**；所有梯度检查通过。"),
        mo.ui.table([
            {
                "advantage": 1 if i == 0 else -1,
                "ratio": ratios[i].tolist(),
                "noclip_loss": unclipped_loss[i].detach().tolist(),
                "grpo_loss": clipped_loss[i].detach().tolist(),
                "noclip_gradient": unclipped_gradient[i].tolist(),
                "grpo_gradient": current_log_probs.grad[i].tolist(),
            }
            for i in range(2)
        ]),
    ])
    return


if __name__ == "__main__":
    app.run()
