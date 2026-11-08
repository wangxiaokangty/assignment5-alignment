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


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Reduce valid-token losses by sequence means or a fixed denominator.

    An empty response contributes zero while retaining its place in the batch.
    Constant normalization divides the total token loss without a batch mean.
    """
    masked_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "constant":
        if normalization_constant is None or normalization_constant <= 0:
            raise ValueError("constant normalization requires a positive normalization_constant")
        return masked_loss.sum() / normalization_constant
    if loss_normalization == "sequence":
        sequence_lengths = mask.sum(dim=-1).clamp_min(1)
        return (masked_loss.sum(dim=-1) / sequence_lengths).mean()
    raise NotImplementedError(loss_normalization)
