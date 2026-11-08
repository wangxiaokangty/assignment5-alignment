import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # think_about_length_normalization

    主作业 PDF §5.1：比较逐序列长度归一化与固定常数归一化。
    记第 $i$ 条回答的长度为 $T_i$，序列优势为 $A_i$，token score 为
    $s_{it}=\nabla_\theta\log\pi_\theta(y_{it}\mid x_i,y_{i,<t})$。
    暂时固定优势计算方式，只比较下面两个梯度估计器：

    $$
    \hat g_{\mathrm{seq}}=\frac1N\sum_{i=1}^N
      \frac{A_i}{T_i}\sum_{t=1}^{T_i}s_{it},\qquad
    \hat g_{\mathrm{const}}=\frac1{NL}\sum_{i=1}^N
      A_i\sum_{t=1}^{T_i}s_{it}.
    $$

    **按序列长度归一化**使每条回答的 token 权重之和相同；长回答不会仅因拥有更多
    token 而在 loss 中占更大权重。长度差异较大、希望每条样本影响相近时，这种方式
    可以改善更新尺度的稳定性。不过它把序列梯度额外乘以随机变量 $1/T_i$，会改变
    原本最大化序列奖励的梯度方向：同样优势下，短回答的每个 token 获得更强的更新。
    对正优势而言倾向于更强地强化短回答；对负优势而言则更强地惩罚短回答，可能使
    错误的长回答相对不易被压制。因此不能把这种归一化简单理解为总是鼓励简短。

    **固定常数归一化**保留完整序列的 score
    $\sum_t s_{it}=\nabla_\theta\log\pi_\theta(y_i\mid x_i)$。
    在 on-policy、无 clipping、无优势标准差缩放，且使用不依赖当前采样回答的
    baseline 时，其期望是序列期望奖励梯度的 $1/L$ 倍；固定 $L$ 只改变全局尺度。
    Dr. GRPO 使用包含自身奖励的组均值 baseline，还会引入 $(G-1)/G$ 的有限组大小
    缩放，所以消除长度归一化不等于自动消除所有估计偏差。
    固定归一化适合奖励只看最终答案、正确推理可能确实需要较长过程的任务。
    代价是长序列拥有更多 token 项，可能导致更大的梯度贡献和方差，并且实际长度远
    小于 $L$ 时梯度整体偏小，需要调整学习率；梯度向量也可能相互抵消，因此其范数
    并不一定随长度线性增长。

    例如，两条同为正确答案的回答长 16 和 128 个 token、优势均为 $+1$，逐序列
    归一化给前者每个 token 的权重是后者的 8 倍，固定归一化则相同。若它们表达
    同样内容、希望每个训练样本权重相当，前者较自然；若长回答是解难题不可缺少的
    推导，而优化目标就是最终正确率，后者更忠实于目标。若长度全部相同，两种方法
    只差一个可以由学习率吸收的常数；若要明确控制计算成本，最好在奖励中表达长度
    成本，而不是依赖长度归一化间接实现。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 一次数值演示：每个 token 的权重

    下面取两条回答、$N=2$、固定最大长度 $L=512$，直接计算梯度公式中乘在每个
    token score 前的系数。表格展示的是权重，不是假设真实模型的梯度方向相同。
    """)
    return


@app.cell
def _(mo):
    mo.ui.table([
        {
            "长度": length,
            "优势": advantage,
            "逐序列：每 token 系数": advantage / (2 * length),
            "固定：每 token 系数": advantage / (2 * 512),
            "逐序列：系数之和": advantage / 2,
            "固定：系数之和": advantage * length / (2 * 512),
        }
        for advantage in (1, -1)
        for length in (16, 128)
    ])
    return


if __name__ == "__main__":
    app.run()
