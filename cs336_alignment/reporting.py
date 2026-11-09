"""Reports from persisted experiment observations, never extrapolated final scores."""

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


def _read_metrics(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A training process may currently be appending the last line.
            if index != len(lines) - 1:
                raise
    return rows


def load_experiments(directories):
    """Return run summaries and observations for label -> directory mappings.

    Configs are found recursively, allowing a learning-rate sweep directory.
    A completed marker AND a finite validation reward at the configured final
    step are necessary to call a run complete. Failure markers take precedence.
    """
    runs, observations = [], []
    for label, directory in directories.items():
        for path in sorted(Path(directory).rglob("config.json")):
            payload = json.loads(path.read_text())
            if "config" not in payload:
                continue
            config = payload["config"]
            metrics = _read_metrics(path.parent / "metrics.jsonl")
            validation = [r for r in metrics if r.get("split") == "val"]
            final = [r for r in validation if r.get("step") == config["steps"]]
            final = final[-1] if final else None
            failure_path = next((path.parent / name for name in ("failure.json", "failed.json")
                                 if (path.parent / name).exists()), None)
            failure = json.loads(failure_path.read_text()) if failure_path else None
            completion_path = path.parent / "completed.json"
            completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
            nonfinite = any(not math.isfinite(r[k]) for r in metrics
                            for k in ("reward", "loss", "grad_norm")
                            if isinstance(r.get(k), (int, float)))
            status = "incomplete"
            if failure is not None:
                status = "diverged" if isinstance(failure, dict) and (
                    failure.get("diverged") is True or failure.get("status") == "diverged"
                ) else "failed"
            elif nonfinite:
                status = "diverged"
            elif completion.get("steps") == config["steps"] and final is not None and math.isfinite(final["reward"]):
                status = "complete"
            train = [r for r in metrics if r.get("split") == "train" and "active_fraction" in r]
            run = dict(config, label=label, directory=str(path.parent), status=status,
                       failure=failure, final_reward=final["reward"] if status == "complete" else None,
                       latest_reward=validation[-1]["reward"] if validation else None,
                       initial_reward=next((r["reward"] for r in validation if r["step"] == 0), None),
                       latest_val_step=validation[-1]["step"] if validation else None,
                       zero_update_fraction=mean(r["active_fraction"] == 0 for r in train) if train else None)
            runs.append(run)
            observations.extend(dict(r, label=label, seed=config["seed"], status=status,
                                     run_id=str(path.parent)) for r in metrics)
    return runs, observations


def summarize_final_rewards(runs, group_by="label", expected_seeds=4, expected_groups=None):
    """Summarize only confirmed final scores; report missing/failed runs separately."""
    groups = sorted(set(expected_groups or []) | {r[group_by] for r in runs})
    summary = []
    for group in groups:
        selected = [r for r in runs if r[group_by] == group]
        complete = [r for r in selected if r["status"] == "complete"]
        values = [r["final_reward"] for r in complete]
        std = stdev(values) if len(values) > 1 else None
        deltas = [r["final_reward"] - r["initial_reward"] for r in complete if r["initial_reward"] is not None]
        zero_fractions = [r["zero_update_fraction"] for r in selected if r["zero_update_fraction"] is not None]
        summary.append(dict(group=group, n_complete=len(values), expected_seeds=expected_seeds,
                            mean=mean(values) if values else None, std=std,
                            variance=std**2 if std is not None else None,
                            min=min(values) if values else None, max=max(values) if values else None,
                            mean_initial_to_final_change=mean(deltas) if deltas else None,
                            n_at_least_25_percent=sum(v >= .25 for v in values),
                            mean_zero_update_fraction=mean(zero_fractions) if zero_fractions else None,
                            status_counts=dict(Counter(r["status"] for r in selected)),
                            missing=max(0, expected_seeds - len(selected))))
    finished = sum(row["n_complete"] for row in summary)
    expected = expected_seeds * len(summary)
    commentary = f"已确认完成 {finished}/{expected} 个计划种子。最终奖励只纳入有完成标记且达到配置终点的验证记录。"
    if not summary or any(row["n_complete"] != expected_seeds for row in summary):
        commentary += " 实验尚未齐全，当前不选择最佳配置，也不能将最新中途奖励当作最终效果。"
    else:
        best = max(summary, key=lambda row: row["mean"])
        ties = [row["group"] for row in summary if row["mean"] == best["mean"]]
        commentary += f" 观测均值最高的配置为 {ties}，均值 {best['mean']:.4f}。"
        comparable_std = [row for row in summary if row["std"] is not None]
        if comparable_std:
            lowest = min(row["std"] for row in comparable_std)
            stable = [row["group"] for row in comparable_std if row["std"] == lowest]
            commentary += f" 观测样本标准差最低的配置为 {stable}，标准差 {lowest:.4f}。"
        commentary += " 四个种子仍是小样本；范围和标准差是描述统计，不能证明差异显著。"
    for row in summary:
        if row["n_complete"]:
            std_text = f"{row['std']:.4f}" if row["std"] is not None else "无法用一个种子估计"
            commentary += f" 配置 {row['group']}：最终均值 {row['mean']:.4f}，样本标准差 {std_text}，达到 25% 的已完成种子为 {row['n_at_least_25_percent']}/{row['n_complete']}。"
            if row["mean_initial_to_final_change"] is not None:
                commentary += f" 初始到最终的平均变化 {row['mean_initial_to_final_change']:+.4f}。"
    if any((row["mean_zero_update_fraction"] or 0) > 0 for row in summary):
        commentary += " 存在全零 advantage 更新：这些批次跳过模型计算，记录的零 entropy 是缺失观测约定，不能解读为确定性策略。"
    commentary += " 失败不等于数值发散；仅明确发散标记或非有限训练数值记作 diverged。反复用验证集调参也会引入选择偏差。"
    return summary, commentary


def plot_final_rewards(runs, group_by="label", expected_seeds=4, expected_groups=None):
    """Plot seed final rewards, their mean and min/max; annotate absent results."""
    summary, _ = summarize_final_rewards(runs, group_by, expected_seeds, expected_groups)
    fig, ax = plt.subplots(figsize=(9, 5))
    numeric = group_by == "learning_rate"
    for index, row in enumerate(summary):
        x = row["group"] if numeric else index
        selected = [r for r in runs if r[group_by] == row["group"] and r["status"] == "complete"]
        for run in selected:
            ax.scatter(x, run["final_reward"], alpha=.7, marker=["o", "s", "^", "D"][run["seed"] % 4],
                       label=f"seed {run['seed']}" if index == 0 else None)
        if selected:
            ax.errorbar(x, row["mean"], yerr=[[row["mean"] - row["min"]], [row["max"] - row["mean"]]],
                        fmt="_", color="black", capsize=6, markersize=18)
        counts = row["status_counts"]
        note = f"{row['n_complete']}/{expected_seeds} complete"
        for status in ("incomplete", "failed", "diverged"):
            if counts.get(status):
                note += f"\n{status}: {counts[status]}"
        ax.text(x, .98, note, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    if numeric:
        ax.set_xscale("log")
        ax.set_xticks([row["group"] for row in summary], [f"{row['group']:g}" for row in summary])
    else:
        ax.set_xticks(range(len(summary)), [str(row["group"]) for row in summary], rotation=15)
    ax.set(xlabel=group_by, ylabel="Final validation reward", ylim=(-.03, 1.15),
           title="Confirmed final scores: seeds, mean and min/max")
    if not any(row["n_complete"] for row in summary):
        ax.text(.5, .45, "No completed final evaluations yet", transform=ax.transAxes, ha="center")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)
    fig.tight_layout()
    return fig


def plot_metric_curves(observations, metric="reward", split="val", x="step"):
    """Show observed runs; shared-step mean/range never extrapolates missing steps.

    For entropy, skipped updates are omitted: zero is a missing-observation
    convention, not evidence that the policy is deterministic.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    selected = [r for r in observations if r.get("split") == split and metric in r and x in r
                and math.isfinite(r[metric])
                and not (metric == "token_entropy" and r.get("active_fraction") == 0)]
    for label in sorted({r["label"] for r in selected}):
        group = [r for r in selected if r["label"] == label]
        run_ids = sorted({r["run_id"] for r in group})
        color = None
        for run_id in run_ids:
            data = sorted((r for r in group if r["run_id"] == run_id), key=lambda r: r[x])
            line, = ax.plot([r[x] for r in data], [r[metric] for r in data], alpha=.4, color=color,
                            label=f"{label} seed {data[0]['seed']} ({data[0]['status']})")
            color = line.get_color()
        if x == "step":
            by_step = {}
            for row in group:
                by_step.setdefault(row[x], {})[row["run_id"]] = row[metric]
            shared = sorted(step for step, values in by_step.items() if len(values) == len(run_ids))
            if shared:
                values = [list(by_step[step].values()) for step in shared]
                ax.plot(shared, [mean(v) for v in values], color=color, label=f"{label} observed-seed mean")
                ax.fill_between(shared, [min(v) for v in values], [max(v) for v in values], color=color, alpha=.15)
    if selected:
        ax.legend(fontsize=7)
    else:
        ax.text(.5, .5, "No observations yet", transform=ax.transAxes, ha="center")
    ax.set(xlabel=x, ylabel=metric, title=f"{split}: {metric} (observed data only)")
    fig.tight_layout()
    return fig


def summarize_runtime(runs, observations):
    """Report measured completed-run costs, including evaluation and checkpoints."""
    rows = []
    for run in runs:
        if run["status"] != "complete":
            continue
        measured = [row for row in observations if row["run_id"] == run["directory"]]
        train = [row for row in measured if row["split"] == "train"]
        marker = json.loads((Path(run["directory"]) / "completed.json").read_text())
        rows.append(dict(
            method=run["label"], seed=run["seed"], final_reward=run["final_reward"],
            total_hours=marker["wall_seconds"] / 3600,
            rollout_seconds=mean(row["rollout_seconds"] for row in train),
            update_seconds=mean(row["update_seconds"] for row in train),
            optimizer_calls=sum(row.get("optimizer_calls", 1) for row in train),
            active_optimizer_calls=sum(row.get("active_optimizer_calls", int(row.get("active_fraction", 1) > 0)) for row in train),
        ))
    return rows


def paired_reward_differences(runs, baseline, candidate):
    """Compare confirmed final rewards only where both methods share a seed."""
    by_method = {
        method: {r["seed"]: r["final_reward"] for r in runs
                 if r["label"] == method and r["status"] == "complete"}
        for method in (baseline, candidate)
    }
    rows = [dict(seed=seed, baseline=by_method[baseline][seed],
                 candidate=by_method[candidate][seed],
                 difference=by_method[candidate][seed] - by_method[baseline][seed])
            for seed in sorted(by_method[baseline].keys() & by_method[candidate].keys())]
    differences = [r["difference"] for r in rows]
    summary = dict(n_pairs=len(rows), mean_difference=mean(differences) if rows else None,
                   sample_std=stdev(differences) if len(rows) > 1 else None)
    return rows, summary
