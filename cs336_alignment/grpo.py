"""Policy updates composed from scoring, reward, and objective primitives."""

import torch

from cs336_alignment.losses import aggregate_loss_across_microbatch, compute_policy_gradient_loss
from cs336_alignment.policy import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.rewards import compute_group_normalized_rewards, compute_rollout_rewards


def grpo_train_step(
    model, tokenizer, optimizer, gradient_accumulation_steps, max_grad_norm,
    reward_fn, repeated_prompts, rollout_responses, repeated_ground_truths, group_size,
    baseline="mean", advantage_eps=1e-6, advantage_normalizer="std",
    importance_reweighting_method="none", old_log_probs=None, cliprange=None,
    loss_normalization="sequence", normalization_constant=None,
):
    """Accumulate a rollout batch, clip once, step once, and clear gradients."""
    if (baseline, advantage_normalizer, importance_reweighting_method, loss_normalization) != (
        "mean", "std", "none", "sequence"
    ):
        raise NotImplementedError("Only standard on-policy GRPO is implemented")
    batch_size = len(repeated_prompts)
    if not 1 <= gradient_accumulation_steps <= batch_size:
        raise ValueError("Accumulation steps must be between one and batch size")
    rewards, metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, reward_stats = compute_group_normalized_rewards(
        rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    metadata.update(reward_stats)
    device = next(model.parameters()).device
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    advantages = advantages.to(device)
    total_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    token_count = 0
    optimizer.zero_grad(set_to_none=True)
    for indices in torch.tensor_split(torch.arange(batch_size), gradient_accumulation_steps):
        micro = {key: value[indices].to(device) for key, value in batch.items()}
        scores = get_response_log_probs(model, micro["input_ids"], micro["labels"], True)
        per_token, _ = compute_policy_gradient_loss(advantages[indices], scores["log_probs"])
        loss = aggregate_loss_across_microbatch(per_token, micro["response_mask"])
        loss = loss * (len(indices) / batch_size)
        loss.backward()
        total_loss += loss.detach()
        entropy_sum += (scores["token_entropy"].detach() * micro["response_mask"]).sum()
        token_count += int(micro["response_mask"].sum())
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm if max_grad_norm is not None else float("inf")
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metadata.update(grad_norm=grad_norm.detach(), token_entropy=entropy_sum / max(token_count, 1))
    return total_loss, metadata
