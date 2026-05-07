---
title: "Activation checkpointing"
description: "Trade compute for memory: drop activations during the forward pass and recompute them during the backward pass. The cheapest way to fit a larger model on the same GPU."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Activation checkpointing (also called gradient checkpointing) saves only a subset of activations during the forward pass and recomputes the rest from those saved checkpoints during the backward pass. Memory drops at the cost of one extra forward pass per checkpoint segment.

## Why it matters

Backprop needs every layer's input activation to compute that layer's parameter gradient. For a deep model the activations dominate training memory. Often more than parameters and optimizer state combined. A 7B-parameter transformer with 32 layers, batch 1, sequence 4096 stores tens of GB of activations.

Checkpointing recovers this memory at a typical cost of ~33% extra training time (one extra forward pass over the checkpointed segments). It is the standard way to fit large transformers on memory-constrained GPUs.

## The mechanism

Partition the model into $K$ segments. During forward:

1. Run the segment.
2. Save **only** its input (the checkpoint).
3. Discard intermediate activations.

During backward:

1. Recompute the segment's forward pass starting from the saved input.
2. Compute gradients normally for that segment.
3. Discard the recomputed activations.

For a transformer, the natural segment is one transformer block. PyTorch provides `torch.utils.checkpoint.checkpoint(...)` and `checkpoint_sequential(...)`; modern training stacks expose this as a single flag (e.g., `gradient_checkpointing=True` in HuggingFace `Trainer`).

## Cost model

- **Memory**: dominant activation memory drops from $O(L)$ to $O(L/K + \sqrt{L})$ activations for $K$ checkpointed segments. For "checkpoint every block" with $L$ blocks, memory drops by ~$L$×.
- **Compute**: each backward step does one extra forward pass per segment. Wall-clock overhead ~33% for typical transformer training (sometimes less because the recomputed forward fuses well with backward kernels).

## When to use

- **Always** when training would OOM otherwise.
- **Selectively** for the most memory-intensive blocks (FFN > attention typically). Selective checkpointing recovers most memory at lower compute cost.
- **Less useful** when peak memory is dominated by optimizer state (use [FSDP / ZeRO](/reference/fsdp-and-zero/) instead).
- **Less useful** at inference (no backward pass).

## Combined with other techniques

- **FSDP**: orthogonal. FSDP shards parameters / gradients / optimizer state; checkpointing reduces activation memory. Most large training runs use both.
- **Mixed precision**: orthogonal; checkpointing saves activations in whatever precision they were computed.
- **CPU offload**: offload activations to CPU memory instead of recomputing. Saves GPU memory at higher communication cost.

## Common pitfalls

- **Recomputing through randomness.** Forward passes with dropout or other stochastic ops must use the same RNG state at recomputation; PyTorch's checkpoint utility handles this with `preserve_rng_state=True` (default).
- **Checkpointing too aggressively.** Checkpointing every layer maximizes memory savings but causes ~50% slowdown; per-block is the sweet spot for transformers.
- **Forgetting that the recomputation runs inside the backward graph.** Custom forward hooks may fire twice; gradients stay correct.
- **Trying to checkpoint inference.** Checkpointing only helps when there is a backward pass to run.

## Related

- [FSDP and ZeRO](/reference/fsdp-and-zero/). For sharding optimizer state and parameters.
- [Mixed precision training](/reference/mixed-precision-training/). Independent memory reduction.
- [Gradient accumulation](/reference/gradient-accumulation/). Simulate larger batches without growing per-step activation memory.
