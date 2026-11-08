"""Policy-gradient objectives and loss reductions."""

from typing import Literal

import torch


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return unreduced negative reward-weighted log probabilities.

    Rewards/advantages are fixed rollout targets, with one value per sequence.
    Response masking and sequence/batch reduction belong to the caller.
    """
    if importance_reweighting_method != "none":
        raise NotImplementedError(importance_reweighting_method)
    advantages = raw_rewards_or_advantages.detach().reshape(-1, 1)
    return -advantages * policy_log_probs, {}
