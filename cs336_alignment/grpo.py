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
    """Accumulate nonzero advantages with the original batch normalization.

    token_entropy measures only active response tokens; active_fraction records
    the retained sequence fraction. All-zero batches skip the optimizer step,
    including momentum and weight decay, and return zero entropy by convention.
    """
    if importance_reweighting_method != "none":
        raise NotImplementedError("Only on-policy updates are implemented")
    if loss_normalization not in ("sequence", "constant"):
        raise ValueError(f"Unknown loss normalization: {loss_normalization}")
    if loss_normalization == "constant" and (
        normalization_constant is None or normalization_constant <= 0
    ):
        raise ValueError("constant normalization requires a positive normalization_constant")
    batch_size = len(repeated_prompts)
    if len(rollout_responses) != batch_size:
        raise ValueError("Prompts and responses must have the same length")
    if not 1 <= gradient_accumulation_steps <= batch_size:
        raise ValueError("Accumulation steps must be between one and batch size")
    rewards, metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, reward_stats = compute_group_normalized_rewards(
        rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    metadata.update(reward_stats)
    device = next(model.parameters()).device
    total_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    token_count = 0
    optimizer.zero_grad(set_to_none=True)
    active_indices = advantages.nonzero(as_tuple=True)[0]
    metadata["active_fraction"] = len(active_indices) / batch_size
    if not len(active_indices):
        metadata.update(grad_norm=total_loss.clone(), token_entropy=total_loss.clone())
        return total_loss, metadata
    batch = tokenize_prompt_and_output(
        [repeated_prompts[i] for i in active_indices.tolist()],
        [rollout_responses[i] for i in active_indices.tolist()], tokenizer,
    )
    advantages = advantages[active_indices].to(device)
    microbatch_size = (batch_size + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    for indices in torch.arange(len(active_indices)).split(microbatch_size):
        micro = {key: value[indices].to(device) for key, value in batch.items()}
        scores = get_response_log_probs(model, micro["input_ids"], micro["labels"], True)
        per_token, _ = compute_policy_gradient_loss(advantages[indices], scores["log_probs"])
        loss = aggregate_loss_across_microbatch(
            per_token, micro["response_mask"], loss_normalization, normalization_constant
        )
        if loss_normalization == "sequence":
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
