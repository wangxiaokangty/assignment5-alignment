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
    # Problem — aggregate_loss_across_microbatch_constant

    Dr. GRPO 的常数归一化为 $L=\sum_{i,t}m_{it}\ell_{it}/Z$，
    不再对每个 response 的长度求平均，也没有额外的 batch 平均。
    固定 $Z$ 后，每个有效 token 的权重都是 $1/Z$，避免长短 response
    因长度归一化而获得不同的 token 权重。

    下面两个 response 分别有 2、3 个有效 token，总损失 24，取 $Z=8$，
    预期 loss 为 3，每个有效 token 梯度为 0.125。最后验证微批次的损失
    可以直接相加；每个微批次使用同一个全局 $Z$。
    """)
    return


@app.cell
def _(aggregate_loss_across_microbatch, torch):
    token_losses = torch.tensor([[100.0, 2.0, 4.0, 100.0], [100.0, 3.0, 6.0, 9.0]], requires_grad=True)
    response_mask = torch.tensor([[False, True, True, False], [False, True, True, True]])
    normalizer = 8
    loss = aggregate_loss_across_microbatch(token_losses, response_mask, "constant", normalizer)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(3.0))
    torch.testing.assert_close(token_losses.grad, response_mask.float() / normalizer)
    microbatch_sum = sum(
        aggregate_loss_across_microbatch(token_losses[i:i + 1], response_mask[i:i + 1], "constant", normalizer)
        for i in range(len(token_losses))
    )
    torch.testing.assert_close(microbatch_sum, loss)
    return loss, microbatch_sum, response_mask, token_losses


@app.cell
def _(loss, microbatch_sum, mo, response_mask, token_losses):
    mo.vstack([
        mo.md(f"实际 loss：**{loss.item():.3f}**；分成两次计算之和：**{microbatch_sum.item():.3f}**。"),
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
