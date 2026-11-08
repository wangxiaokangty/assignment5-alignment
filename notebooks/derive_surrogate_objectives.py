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
# Problem — derive_surrogate_objectives

为简化记号固定 prompt $x$，假设 $L$ 为偶数，旧策略 $\pi_0$ 固定且在新策略支持集上非零。
令 $k=2t-1$，构造混合策略
$$q_{\theta,t}(y|x)=
\left[\prod_{u<k}\pi_0(y_u|x,y_{<u})\right]
\pi_\theta(y_k|x,y_{<k})\pi_\theta(y_{k+1}|x,y_{<k+1})
\left[\prod_{u>k+1}\pi_0(y_u|x,y_{<u})\right].$$
即 prefix 用旧策略，连续两个 token 用当前策略，suffix 再用旧策略。
后缀条件包含刚生成的两个新 token，每个条件分布均归一化，因此 $q$ 是合法分布。

所求 surrogate 为
$$\widetilde J(\theta)=\sum_{t=1}^{L/2}\mathbb E_{y\sim q_{\theta,t}(\cdot|x)}[r(y|x)].$$
若有 prompt 分布，再在外层取 $\mathbb E_{x\sim\rho}$；题目给的是求和而非均值，不加 $2/L$。

记 $w_t=q_{\theta,t}/\pi_0$，消去共有的 prefix/suffix 因子得到
$$w_t=\frac{\pi_\theta(y_{2t-1}|x,y_{<2t-1})\pi_\theta(y_{2t}|x,y_{<2t})}
{\pi_0(y_{2t-1}|x,y_{<2t-1})\pi_0(y_{2t}|x,y_{<2t})}.$$
故
$$\nabla_\theta\widetilde J
=\mathbb E_{y\sim\pi_0}\!\left[\sum_t r(y|x)\nabla_\theta w_t\right]
=\mathbb E_{y\sim\pi_0}\!\left[\sum_t w_t r(y|x)
\nabla_\theta\log\left(\pi_\theta(y_{2t-1}|x,y_{<2t-1})\pi_\theta(y_{2t}|x,y_{<2t})\right)\right].$$
这正是题目估计器的期望。它校正每对内部两 token 的联合分布，
仍未校正 pair 以外的 prefix 和 suffix；一般不等于完整新策略期望奖励的梯度。
$L=2$ 时仅一对，混合策略退化为完整新策略，恰好恢复正确的序列级重加权。
若 $L$ 为奇数，可另加最后单 token 的混合策略项。
""")
    return


if __name__ == "__main__":
    app.run()
