"""Reward evaluation for policy rollouts."""

from collections.abc import Callable
from typing import Literal

import torch


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Score each response and return CPU float32 rewards and component means."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("Responses and ground truths must have the same length")
    if not rollout_responses:
        raise ValueError("The rollout batch must not be empty")

    scores = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor([score["reward"] for score in scores], dtype=torch.float32)
    metadata = {
        f"mean_{component}": sum(score[component] for score in scores) / len(scores)
        for component in ("reward", "format_reward", "answer_reward")
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply a group baseline and optional standard-deviation or mean scaling."""
    if baseline not in ("mean", "none"):
        raise ValueError(f"Unknown baseline: {baseline}")
    if advantage_normalizer not in ("std", "none", "mean"):
        raise ValueError(f"Unknown advantage normalizer: {advantage_normalizer}")
    if raw_rewards.ndim != 1 or raw_rewards.numel() == 0:
        raise ValueError("Rewards must be a nonempty one-dimensional tensor")
    if group_size < 1 or raw_rewards.numel() % group_size:
        raise ValueError("group_size must be positive and divide the rollout batch size")
    if advantage_eps <= 0:
        raise ValueError("advantage_eps must be positive")

    grouped = raw_rewards.reshape(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    # A singleton group has zero variation (and zero mean-centered advantage).
    stds = grouped.std(dim=1, keepdim=True, correction=1 if group_size > 1 else 0)
    advantages = grouped - means if baseline == "mean" else grouped
    if advantage_normalizer == "std":
        advantages = advantages / (stds + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (means + advantage_eps)
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_group_std": stds.mean().item(),
    }
    return advantages.reshape_as(raw_rewards), metadata
