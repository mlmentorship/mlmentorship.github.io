import torch


def next_token_batch(tokens: torch.Tensor, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build causal-LM inputs and labels from a padded token batch."""
    if tokens.ndim != 2 or tokens.size(1) < 2:
        raise ValueError("tokens must have shape [batch, sequence] with sequence >= 2")
    inputs = tokens[:, :-1]
    # BUG: a causal LM predicts the following token, not the current token.
    targets = tokens[:, :-1].clone()
    # BUG: padded labels must use cross-entropy's ignore index.
    targets[targets == pad_id] = pad_id
    return inputs, targets
