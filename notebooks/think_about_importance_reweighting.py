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
# Problem — think_about_importance_reweighting

**不重加权** 把旧策略产生的数据当成当前策略数据来更新，通常避免了重要性比率引起的方差；
但对当前期望奖励梯度有分布偏移偏差。当策略刚采完样本、更新很少或学习率小，偏移可能可接受。
若反复更新同一批数据，忽略偏移的风险增加。

**PPO/GRPO token 比率加 clipping** 校正单个位置的动作概率，
而 prefix 和 suffix 仍来自旧策略，因此即使不 clipping 也有 surrogate 偏差。
clipping 进一步引入偏差，但能限制使已有优势方向更新过猛的 token 项，
在较长响应、希望局部控制更新时有用。裁剪不是严格的 KL 或信赖域约束，
共享参数更新仍可能使策略变化很大。

**GSPO 序列几何均值比率加 clipping** 用响应 token 的平均 log-ratio 再取 exp，
同一回答共享一个裁剪决策；相比完整序列比率的连乘通常更稳定，避免方差随长度剧增。
它会将局部的高低比率相互抵消，也可能因一个序列被裁剪而同时丢弃全部 token 的信号。
几何均值不是新旧序列概率的精确比率，所以虽然使用序列统计，**并非无偏序列重要性采样**。
当奖励反映整条回答质量、token 级比率噪声大时可能有优势；
当少数关键 token 需要单独约束时，token clipping 更直观。

这是偏差、方差及粒度的权衡，不存在对所有模型和奖励都成立的三者严格排序。
无裁剪的完整序列重要性采样在支持集条件下无偏，但长序列方差可能极大；
题目中的三种方法都以不同方式牺牲精确校正来获得可用的训练信号。
应实测验证正确率、种子方差、梯度/熵、clip fraction 和耗时，而不能只比较训练 loss。
""")
    return


if __name__ == "__main__":
    app.run()
