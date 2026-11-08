import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch

    from cs336_alignment.losses import aggregate_loss_across_microbatch

    return aggregate_loss_across_microbatch, mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Problem — aggregate_loss_across_microbatch_sequence

    每条 response 先对有效 token 求平均，再对 batch 内各条 response 求平均：
    $L=\frac1B\sum_i\frac{\sum_t m_{it}\ell_{it}}{\sum_t m_{it}}$。
    因此长短 response 的总权重相同。下面两个 response 长度分别为 2 和 3，
    损失分别为 3 和 6，所以最终 loss 应为 4.5。
    被 mask 排除的位置梯度必须为零；有效 token 梯度分别为 1/4 和 1/6。
    为避免零长度 response 的除零，核心实现将其平均损失定义为零。
    """)
    return


@app.cell
def _(aggregate_loss_across_microbatch, torch):
    token_losses = torch.tensor([[100.0, 2.0, 4.0, 100.0], [100.0, 3.0, 6.0, 9.0]], requires_grad=True)
    response_mask = torch.tensor([[False, True, True, False], [False, True, True, True]])
    loss = aggregate_loss_across_microbatch(token_losses, response_mask)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(4.5))
    torch.testing.assert_close(
        token_losses.grad,
        torch.tensor([[0.0, 0.25, 0.25, 0.0], [0.0, 1 / 6, 1 / 6, 1 / 6]]),
    )
    empty_loss = aggregate_loss_across_microbatch(token_losses, torch.zeros_like(response_mask))
    torch.testing.assert_close(empty_loss, torch.tensor(0.0))
    return loss, response_mask, token_losses


@app.cell
def _(loss, mo, response_mask, token_losses):
    mo.vstack([
        mo.md(f"实际聚合损失：**{loss.item():.2f}**；数值、mask 梯度和空 response 检查通过。"),
        mo.ui.table([
            {
                "sequence": i,
                "token_losses": token_losses[i].detach().tolist(),
                "response_mask": response_mask[i].tolist(),
                "gradient": token_losses.grad[i].tolist(),
            }
            for i in range(len(token_losses))
        ]),
    ])
    return


if __name__ == "__main__":
    app.run()
