---
title: "Activation checkpointing"
description: "Trade compute for memory: drop activations during the forward pass and recompute them during the backward pass. The cheapest way to fit a larger model on the same GPU."
date: "2026-05-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Activation checkpointing (also called gradient checkpointing) saves only a subset of activations during the forward pass and recomputes the rest from those saved checkpoints during the backward pass. Memory drops at the cost of one extra forward pass per checkpoint segment.

Backprop needs every layer's input activation to compute that layer's parameter gradient. For a deep model the activations dominate training memory. Often more than parameters and optimizer state combined. A 7B-parameter transformer with 32 layers, batch 1, sequence 4096 stores tens of GB of activations.

Checkpointing recovers this memory by repeating part of the forward work. Recomputing the full forward graph adds about one forward pass of FLOPs. Wall-time cost depends on kernels, memory traffic, and how much of the graph is recomputed.

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

- **Memory across a simple chain**: if each segment has $K$ layers, peak saved boundary state scales like $O(L/K)$ and peak temporary recomputation state like $O(K)$. The total $O(L/K + K)$ is minimized near $K=\sqrt{L}$.
- **Memory inside a transformer block**: checkpointing every block still stores block boundaries, but discards the larger internal matrix and attention intermediates. The reduction is a workload-dependent constant factor, not $L$×.
- **Compute**: full rematerialization adds roughly one forward pass to a training step. Since a forward and backward step is often estimated at three forward-pass units, this is about 33% more FLOPs. Wall-time overhead can differ.

## When to use

- **Always** when training would OOM otherwise.
- **Selectively** for the most memory-intensive blocks (FFN > attention typically). Selective checkpointing recovers most memory at lower compute cost.
- **Less useful** when peak memory is dominated by optimizer state (use [FSDP / ZeRO](/concepts/fsdp-and-zero/) instead).
- **Less useful** at inference (no backward pass).

## Combined with other techniques

- **FSDP**: orthogonal. FSDP shards parameters / gradients / optimizer state; checkpointing reduces activation memory. Most large training runs use both.
- **Mixed precision**: orthogonal; checkpointing saves activations in whatever precision they were computed.
- **CPU offload**: offload activations to CPU memory instead of recomputing. Saves GPU memory at higher communication cost.

## Common pitfalls

- **Recomputing through randomness.** Forward passes with dropout or other stochastic ops must use the same RNG state at recomputation; PyTorch's checkpoint utility handles this with `preserve_rng_state=True` (default).
- **Checkpointing too aggressively.** Larger rematerialized regions save more boundary state and repeat more work. Profile selective, per-block, and larger-segment choices under the real memory limit.
- **Forgetting that the recomputation runs inside the backward graph.** Custom forward hooks may fire twice; gradients stay correct.
- **Trying to checkpoint inference.** Checkpointing only helps when there is a backward pass to run.

## Related

- [FSDP and ZeRO](/concepts/fsdp-and-zero/). For sharding optimizer state and parameters.
- [Mixed precision training](/concepts/mixed-precision-training/). Independent memory reduction.
- [Gradient accumulation](/concepts/gradient-accumulation/). Simulate larger batches without growing per-step activation memory.
