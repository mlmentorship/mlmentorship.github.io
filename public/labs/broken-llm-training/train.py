from collections.abc import Iterable

import torch
from torch import nn

from tiny_lm.data import next_token_batch


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    pad_id: int,
    accumulation_steps: int = 4,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0
    for tokens in batches:
        # BUG: this clears gradients on every micro-batch, defeating accumulation.
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = next_token_batch(tokens, pad_id)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        (loss / accumulation_steps).backward()
        # BUG: clipping and stepping happen per micro-batch rather than per window.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # BUG: scheduler cadence should follow optimizer updates, not micro-batches.
        scheduler.step()
        total_loss += float(loss.detach())
        batch_count += 1
    return total_loss / max(batch_count, 1)


def evaluate(model: nn.Module, batches: Iterable[torch.Tensor], *, pad_id: int) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    # BUG: evaluation builds computation graphs and never restores prior mode.
    for tokens in batches:
        inputs, targets = next_token_batch(tokens, pad_id)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        total_loss += float(loss)
        batch_count += 1
    return total_loss / max(batch_count, 1)
