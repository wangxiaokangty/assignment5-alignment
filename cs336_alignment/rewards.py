"""Reward evaluation for policy rollouts."""

from collections.abc import Callable

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
