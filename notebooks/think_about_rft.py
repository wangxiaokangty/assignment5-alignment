import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
# Problem — think_about_rft

记单个 prompt 的正确率为 $\eta$，$s_j=\nabla_\theta\log\pi_\theta(y_j|x)$，
$g=\mathbb E[r_js_j]=\nabla_\theta\eta$。以下将采样出的回答、奖励和 baseline 视为常量，
求的是更新 surrogate 的梯度，不是对采样分布再求一次导数。

**RFT。** 二元奖励满足 $\mathbf1\{r_j=1\}=r_j$，故
$\mathbb E[Z^{-1}\sum_j r_js_j]=(G/Z)g$。
只要样本来自当前策略，且分母固定，这是期望奖励梯度的固定倍数；
“仅对正确答案做监督学习”本身也是一种无 baseline 的 REINFORCE 更新。
若改成除以随机的正确样本数量，或对旧样本重复训练，则不能直接沿用该结论。

**Dr. GRPO。** 组均值 $\bar r=G^{-1}\sum_k r_k$ 包含样本自身。
利用 $\mathbb E[s_j]=0$ 和不同样本独立，得到
$\mathbb E[\bar r s_j]=g/G$，从而
$$\mathbb E\left[\frac1Z\sum_j(r_j-\bar r)s_j\right]=\frac{G-1}{Z}g.$$
因此有限 $G$ 时两者期望**并不完全相同**；Dr. GRPO 多出 $(G-1)/G$ 因子。
用 $G/(G-1)$ 校正，或使用 leave-one-out baseline，才恢复相同期望。
$G=1$ 时组均值法梯度恒为零。取 $Z=G$、$G\to\infty$ 时两者都趋于 $g$。

**方差和适用场景。** baseline 常能减少由奖励均值造成的随机性，
并用失败样本主动压低错误回答的概率，因而对混合成功/失败的组通常有益；
但 reward mean 并非总是使梯度方差最小的 baseline，有限样本估计也会引入噪声，
不能保证 Dr. GRPO 方差总更低。比较方差还应先校正上述期望缩放。
RFT 在成功稀少时只需反传少数正确轨迹，算力利用有优势；
Dr. GRPO 在全失败或全成功组都不给梯度，RFT 仍会强化全成功组的采样回答。
例如 $G=1$ 时 RFT 能学习而组均值法不能；成功概率很高且无需继续强化时，
组均值 baseline 可以减少无用更新。下面的 Bernoulli 一次性模拟展示均值与方差，
不把单个例子当成一般性方差排序。
""")
    return


@app.cell
def _(mo, np):
    rng = np.random.default_rng(42)
    rows = []
    for p in (0.1, 0.5, 0.9):
        samples = rng.binomial(1, p, size=(100000, 8))
        score = samples - p
        rft = (samples * score).mean(axis=1)
        dr = ((samples - samples.mean(axis=1, keepdims=True)) * score).mean(axis=1) * 8 / 7
        rows.append({"p": p, "true_gradient": p * (1-p),
                     "rft_mean": rft.mean(), "corrected_dr_mean": dr.mean(),
                     "rft_variance": rft.var(), "corrected_dr_variance": dr.var()})
    mo.ui.table(rows)
    return


if __name__ == "__main__":
    app.run()
