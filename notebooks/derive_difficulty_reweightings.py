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
# Problem — derive_difficulty_reweightings

设 $\eta(x)=\mathbb E_{y\sim\pi_\theta}[r(y|x)]$，$s(y,x)=\nabla_\theta\log\pi_\theta(y|x)$。
二元奖励给出 $\operatorname{Var}(r|x)=\eta(x)(1-\eta(x))$。
在策略可微、可交换期望与梯度的条件下，
$\mathbb E[s|x]=0$，$\mathbb E[rs|x]=\nabla_\theta\eta(x)$。
取题设 $Z=G$，当 $G\to\infty$ 时组均值收敛到 $\eta$，样本标准差收敛到
$\sqrt{\eta(1-\eta)}$。对非退化正确率 $0<\eta<1$，得到：

**(a) Dr. GRPO：$w(x)=1$。**
$$\lim_{G\to\infty}\frac1G\sum_j(r_j-\bar r)s_j
=\mathbb E[(r-\eta)s|x]=\nabla_\theta\eta(x).$$
对 $x\sim\rho$ 再取期望，即原始平均正确率梯度。
有限 $G$ 的组均值 baseline 带来 $(G-1)/G$ 缩放，但极限中消失。

**(b) GRPO（固定长度归一化）：$w(x)=1/\sqrt{\eta(x)(1-\eta(x))}$。**
$$\mathbb E\!\left[\frac{r-\eta}{\sqrt{\eta(1-\eta)}}s\mid x\right]
=\frac{\nabla_\theta\eta(x)}{\sqrt{\eta(x)(1-\eta(x))}}.$$
相比中等难度，极难与极易 prompt 的权重均较大。
此处按题意使用固定常数聚合，不能直接将推导移用到依赖回答长度的聚合。

**(c) MaxRL：$w(x)=1/\eta(x)$。**
$$\mathbb E\!\left[\frac{r-\eta}{\eta}s\mid x\right]
=\frac{\nabla_\theta\eta(x)}{\eta(x)}.$$
因此它相对更强调低正确率 prompt；内部点上也等于 $\nabla_\theta\log\eta(x)$。

三项对应 $J_{\theta,w}=\mathbb E_x[w(x,\mathrm{stopgrad}(\pi_\theta))\eta_\theta(x)]$，
**权重必须停止梯度**，否则会增加 $\eta\nabla_\theta w$ 项。
如保留实现中的 $\epsilon$，两种归一化权重分别为
$1/(\sqrt{\eta(1-\eta)}+\epsilon)$ 与 $1/(\eta+\epsilon)$。
$\eta=0$ 或 $1$ 时不能直接使用奇异的无 epsilon 公式；有限组全相同奖励的实际优势为零，
大权重不等于算法可以从从未采到成功的组中创造学习信号。
""")
    return


if __name__ == "__main__":
    app.run()
