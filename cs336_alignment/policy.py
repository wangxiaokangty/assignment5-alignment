"""Tokenization and differentiable scoring primitives for policy training."""

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Right-pad separately encoded prompts/responses, then shift labels and masks."""
    if not prompt_strs or len(prompt_strs) != len(output_strs):
        raise ValueError("Expected nonempty, equally sized prompt and output batches")
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a pad token")

    prompts = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    responses = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses)]
    if min(lengths) < 1:
        raise ValueError("Each prompt/response pair must contain at least one token")
    tokens = torch.full((len(prompts), max(lengths)), tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        tokens[i, : lengths[i]] = torch.tensor(prompt + response, dtype=torch.long)
        response_mask[i, len(prompt) : lengths[i]] = True
    return {
        "input_ids": tokens[:, :-1],
        "labels": tokens[:, 1:],
        "response_mask": response_mask[:, 1:],
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Score already-shifted labels, preserving gradients and the model's mode.

    Returns scores for every position; callers apply the response mask. Compute
    log-softmax in float32 for half-precision models to stabilize reductions.
    """
    logits = model(input_ids).logits
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    log_probs = logits.log_softmax(dim=-1)
    result = {"log_probs": log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)}
    if return_token_entropy:
        result["token_entropy"] = -(log_probs.exp() * log_probs).sum(dim=-1)
    return result
