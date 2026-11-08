import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from cs336_alignment.baselines import compare_bernoulli_baselines

    return compare_bernoulli_baselines, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Problem 2 — baseline_calcs

    题目：主作业 PDF §4.1.5，印刷页 10–11。令 $A\sim\mathrm{Bernoulli}(p)$、
    $p=\sigma(\theta)$，奖励 $r(A)=A$。以下 $n$ 个动作独立同分布。

    先求 score function。由 $\frac{dp}{d\theta}=p(1-p)$，有
    $$
    \log\pi_\theta(A)=A\log p+(1-A)\log(1-p),\qquad
    \nabla_\theta\log\pi_\theta(A)=A-p.
    $$
    因此真实梯度是 $\nabla_\theta J=\nabla_\theta p=p(1-p)$。

    ## (a) 不使用 baseline

    单个样本的梯度贡献 $Z=A(A-p)$ 在 $A=1$ 时为 $1-p$，在 $A=0$ 时为 $0$。
    所以
    $$
    \mathbb E[Z]=p(1-p),\qquad \mathbb E[Z^2]=p(1-p)^2,
    $$
    $$
    \operatorname{Var}(Z)=p(1-p)^2-p^2(1-p)^2=p(1-p)^3.
    $$
    独立样本平均的方差为单个样本方差除以 $n$：
    $$\boxed{\operatorname{Var}(\hat g_0)=\frac{p(1-p)^3}{n}.}$$

    ## (b) 使用固定 baseline $b$

    令 $Z_b=(A-b)(A-p)$。两个可能的取值为
    $$
    Z_b=\begin{cases}(1-b)(1-p),& A=1\quad(\text{概率 }p),\\
    bp,& A=0\quad(\text{概率 }1-p).
    \end{cases}
    $$
    期望仍然是
    $$\mathbb E[Z_b]=p(1-b)(1-p)+(1-p)bp=p(1-p),$$
    二阶矩为
    $$\mathbb E[Z_b^2]=p(1-b)^2(1-p)^2+(1-p)b^2p^2.$$
    两点分布的方差为 $p(1-p)$ 乘以两个取值之差的平方；这里差值为
    $(1-b)(1-p)-bp=1-p-b$。因此
    $$\boxed{\operatorname{Var}(\hat g_b)=\frac{p(1-p)(1-p-b)^2}{n}.}$$

    对 $0<p<1$，相对于不使用 baseline，方差严格降低当且仅当
    $|1-p-b|<1-p$，即 $0<b<2(1-p)$；边界处相等，区间外反而更高。
    最优 baseline 是 $b^*=1-p$，此时两个动作的梯度贡献都等于 $p(1-p)$，方差为零。
    这是本题二元策略和奖励的特殊性质，不意味着一般任务都能实现零方差。

    $b$ 可以依赖当前策略的 $p$，但在估计器中作为固定系数使用，不对它反向传播。
    本题也没有用当前这批动作的样本均值来替代 $b$。

    ## (c) 总体均值 baseline $b=p$

    代入 (b) 得到
    $$\boxed{\operatorname{Var}(\hat g_p)=\frac{p(1-p)(1-2p)^2}{n}.}$$
    与 (a) 相减：
    $$
    \operatorname{Var}(\hat g_p)-\operatorname{Var}(\hat g_0)
    =\frac{p(1-p)}{n}\left[(1-2p)^2-(1-p)^2\right]
    =\frac{p^2(1-p)(3p-2)}{n}.
    $$

    - $0<p<2/3$：均值 baseline 方差更低。
    - $p=2/3$：两者方差相同。
    - $2/3<p<1$：均值 baseline 方差更高。

    特别地，$p=1/2$ 时均值 baseline 恰好最优，方差为零。
    有限 $\theta$ 对应 $0<p<1$；在退化边界 $p=0,1$（或其极限），两者方差均为零。
    因而**减去平均奖励并不保证降低方差**：本题最优值为 $1-p$，而非一般的 $p$。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 数值验证

    下表每个 $p$ 独立模拟 100,000 批，每批 $n=32$ 个动作，固定 seed=0。
    直接采样成功次数 $K\sim\mathrm{Binomial}(n,p)$，计算
    $\hat g_b=bp+(K/n)(1-p-b)$；这与逐个采样动作再求平均完全等价。
    同一 $p$ 的三种 baseline 共用动作样本，模拟方差使用 `ddof=1`。
    计算在本地 CPU 上完成，无需模型、网络或 GPU。
    """)
    return


@app.cell
def _(compare_bernoulli_baselines, mo):
    comparison = [
        row
        for probability in (0.2, 0.5, 2 / 3, 0.8)
        for row in compare_bernoulli_baselines(probability)
    ]
    mo.ui.table(comparison, page_size=12)
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(0.01, 0.99, step=0.01, value=0.8, label="动作 1 的概率 p", show_value=True)
    n_slider = mo.ui.slider(1, 256, value=32, label="每批样本数 n", show_value=True)
    mo.hstack([p_slider, n_slider])
    return n_slider, p_slider


@app.cell
def _(compare_bernoulli_baselines, mo, n_slider, p_slider):
    mo.ui.table(compare_bernoulli_baselines(p_slider.value, n=n_slider.value))
    return


if __name__ == "__main__":
    app.run()
