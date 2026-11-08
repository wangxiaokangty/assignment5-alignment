# CS336 Spring 2026 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).


## 本地实现与运行

主 PDF 的 28 个 problem 分别对应 `notebooks/<problem_name>.py`，均为 marimo
notebook。核心实现位于 `cs336_alignment/`，`tests/adapters.py` 仅转发调用。
模型默认从 `models/OLMo-2-0425-1B` 加载；训练使用本地 GPU，不依赖 Modal。
本目录的 safety/RLHF PDF 为独立 optional supplement，主作业测试入口如下。

```bash
# 训练使用 PyTorch SDPA，无需安装 flash-attn。
uv sync --extra gpu --no-install-package flash-attn
uv run --no-sync pytest tests/test_grpo.py
uv run --no-sync marimo edit notebooks/grpo_experiments_standard_on_policy.py
```

单个标准实验（200 rollout steps、1024 validation examples）：

```bash
uv run --no-sync python scripts/train_grpo.py --seeds 0 \
  --policy-device cuda:0 --inference-gpu 1 --port 8000
```

`--policy-device` 相对于训练进程的 `CUDA_VISIBLE_DEVICES`；`--inference-gpu`
是物理 GPU 编号。完整矩阵调度器清除子进程的设备可见性限制，按固定物理 GPU 对分配：

```bash
uv run --no-sync python scripts/run_experiments.py --dry-run
uv run --no-sync python scripts/run_experiments.py --pairs 0,1 2,3 4,5 6,7
```

完整矩阵包含标准 GRPO、两档额外学习率、两个额外 prompt、四个 on-policy
变体、四个 off-policy 变体、LOO baseline，各 4 seeds，共 56 次训练。
`--only` 可按输出目录名选择实验。已在外部运行的任务必须使用
`--exclude-run results/grpo_standard/seed_0` 等参数排除，避免重复运行。

每个 `results/.../seed_N/` 保存配置及输入指纹、实际指标、采样回答、滚动
`resume.pt` 和最终模型。`completed.json` 只在完整训练成功后生成；
队列失败写入 `failure.json`，不会以失败记录冒充最终分数。
手动恢复使用原参数加 `--resume`；配置或数据变化会拒绝恢复。
实验报告只将完整种子纳入最终统计，并保留中途曲线供检查。

零优势回答在前向前剪除，所有损失仍按原始 batch 分母归一化。
`token_entropy` 仅针对实际参与更新的 response tokens；全零批次的熵占位值0
不代表策略熵为0，需结合 `active_fraction` 阅读。Off-policy 日志每个 rollout
汇总32次更新，`step` 表示 rollout step；不能当作32次优化器更新的单步计数。

结果和模型默认不进入 Git；重跑不会覆盖已有运行。实验 notebook 可直接读取
本地结果生成表格和曲线，短 smoke 的结果目录独立于正式实验。

本次执行的完整实验队列另有完成检查器持续运行：进度见
`results/lab_progress.json`。它在各题所需的完整 seeds 全部成功后导出报告、保存小型
结果证据并生成该题的独立 commit；原始 rollouts 和模型仍保留在本地。
`status: complete` 还要求主实验四种子平均验证正确率至少25%，以及主作业测试通过。
若训练失败、源文件被另行修改或 Git 暂存区存在其他工作，它会保留现场并标记待检查，
不会将这些情况当作完成。完整 HTML 报告位于 `results/<problem_name>/report.html`。

2026-11-08 中断恢复后，队列和检查器改由本机 systemd 临时服务管理，
不依赖 Codex 临时命令会话的存活（机器重启后需要重新启动服务）：

```bash
systemctl status alignment5-training alignment5-finalizer --no-pager
cat results/lab_progress.json
tail -f results/experiment_queue.log
```

四个标准种子分别从70、50、50、50步检查点恢复；未保存的尾部步骤会重做。
后台服务执行的是既定脚本，不会自动唤醒对话中的 AI 来诊断新的失败。
