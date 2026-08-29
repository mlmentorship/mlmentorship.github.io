---
title: "Gradient accumulation"
description: "Run several forward-backward passes before each optimizer step to simulate a larger effective batch size without the memory cost."
date: "2026-02-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Gradient accumulation runs $K$ forward-backward passes on $K$ micro-batches, summing (or averaging) the gradients across those passes, and then performs a single optimizer step. The effective batch size is $K \times$ the per-pass batch size, with no extra activation memory.

Many training recipes prescribe a specific effective batch size (e.g., 1024 sequences) for stable convergence. If your GPU can only hold 32 sequences at a time, you have two choices: spread the batch across 32 GPUs, or accumulate gradients over 32 steps on one GPU.

Gradient accumulation is the cheap option. It is the standard way to (a) match published training recipes on smaller hardware and (b) increase effective batch size in fine-tuning loops where multi-GPU is not available.

## The mechanism

Replace this loop:

```
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

With:

```
for step, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps   # average over K micro-batches
    loss.backward()                             # gradients accumulate in .grad
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Key points:

- `loss.backward()` adds to existing `.grad` (does not overwrite). Calling it $K$ times accumulates.
- Divide the loss by $K$ so the accumulated gradient equals the average over the effective batch (matches a single big batch's gradient).
- Optimizer step happens once per $K$ micro-batches.

## Cost model

- **Activation memory**: same as one micro-batch (each is forward-backward'd independently).
- **Optimizer memory**: unchanged (no extra optimizer state).
- **Wall clock**: an optimizer step runs $K$ forward-backward micro-batches. Per-sample throughput may stay similar, improve through less frequent communication, or fall because smaller matrix shapes use the accelerator poorly. Measure it.
- **Convergence**: nearly equivalent to a true large-batch step, modulo BatchNorm (see pitfalls).

## Combined with other techniques

- **DDP / FSDP**: gradient accumulation reduces the frequency of inter-GPU communication. With $N$ GPUs and $K$ accumulation steps, only one all-reduce per $K$ micro-batches → faster throughput. PyTorch's `model.no_sync()` skips the all-reduce on intermediate steps.
- **Mixed precision**: works identically; the loss scaler handles accumulated gradients.
- **Activation checkpointing**: orthogonal; combine for maximum effective batch size on minimum memory.

## Common pitfalls

- **Forgetting to divide by $K$.** Without normalization, the gradient magnitude is $K$× larger than expected → effectively a $K$× larger LR.
- **Calling `optimizer.step()` every micro-batch.** Defeats the purpose.
- **Mixing with BatchNorm.** BN computes statistics within a single forward pass. With accumulation, BN sees only the micro-batch. Statistics are noisier than at the effective batch size. Use LayerNorm or GroupNorm instead, or sync BN across micro-batches.
- **Using accumulation as a replacement for distributed training when bandwidth is available.** Multi-GPU with proper sharding is faster and more memory-efficient than serial accumulation.

## Related

- [Activation checkpointing](/concepts/activation-checkpointing/). Independent memory reduction.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Distributed memory reduction.
- [Mixed precision training](/concepts/mixed-precision-training/). Orthogonal memory reduction.
