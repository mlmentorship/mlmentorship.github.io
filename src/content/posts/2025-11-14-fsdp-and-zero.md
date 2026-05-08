---
title: "FSDP and ZeRO: sharding optimizer state, gradients, and parameters"
description: "How modern training scales beyond a single GPU's memory by partitioning the optimizer state, gradients, and parameters across the data-parallel group."
date: "2025-11-14"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Fully Sharded Data Parallel (FSDP) and ZeRO partition the optimizer state, gradients, and (optionally) the model parameters across the data-parallel GPUs, so each GPU holds only $1/N$ of each, instead of every GPU holding the full copy.

## Why it matters

Training memory has four big consumers:

| Component | Bytes per parameter (BF16 + FP32 master + Adam) |
|-----------|----------------------|
| Parameters (BF16) | 2 |
| Gradients (BF16) | 2 |
| Adam first moment $m$ (FP32) | 4 |
| Adam second moment $v$ (FP32) | 4 |
| Master weights (FP32, optional) | 4 |
| **Total** | **~16 bytes/param** |

A 7B-parameter model needs ~112 GB of training state. Too much for one 80 GB H100. Sharding spreads it across $N$ GPUs so each holds ~$112 / N$ GB. With $N = 8$: ~14 GB per GPU.

ZeRO ([Rajbhandari et al., 2019](https://arxiv.org/abs/1910.02054), DeepSpeed) and PyTorch FSDP implement this idea. They are the standard for any training run that doesn't fit in a single GPU's memory and doesn't need full tensor or pipeline parallelism.

## The three stages (ZeRO-1/2/3)

### Stage 1: shard optimizer state
Each GPU holds the full parameters and gradients but only $1/N$ of the Adam moments. After the backward pass, each GPU updates its slice and then all-gathers updated parameters.
Memory reduction: ~4× for Adam.

### Stage 2: shard optimizer state + gradients
Same as Stage 1 plus gradients are reduced-scattered (each GPU keeps its slice) instead of all-reduced.
Memory reduction: ~8×.

### Stage 3 (FSDP): shard optimizer state + gradients + parameters
Each GPU holds only its slice of the parameters. Before each forward pass through a layer, all-gather the parameters; immediately free them after the layer's backward pass.
Memory reduction: nearly $N$×. Communication overhead: an all-gather per forward and per backward layer.

PyTorch FSDP is the standard implementation of Stage 3 and is what most modern open-source training (Llama, Mistral, etc.) uses.

## Tradeoffs

- **Memory vs. communication**: each stage trades more communication for less memory.
- **Sharding granularity**: FSDP can wrap individual layers ("auto-wrap policy") so all-gathers cover only one layer's parameters at a time, capping peak unsharded memory.
- **Mixing with tensor parallelism**: FSDP shards across the data-parallel dimension; for very large models, combine with tensor parallelism (TP) within a node and FSDP across nodes (3D parallelism).

## When to use what

| Model size | Strategy |
|-----------|----------|
| Fits on 1 GPU | DDP (no sharding) |
| 7B–70B on 8–64 GPUs | FSDP / ZeRO-3 |
| 70B+ across multi-node | FSDP + tensor parallelism |
| 175B+ | FSDP + TP + pipeline parallelism (3D) |

## Common pitfalls

- **Activation memory is not sharded.** FSDP shards parameters and optimizer state but each GPU still holds the activations for its data-parallel slice. Use [activation checkpointing](/concepts/activation-checkpointing/) to reduce that.
- **All-gather overhead at small layer size.** Wrapping every linear layer separately can dominate runtime; wrap at transformer-block granularity instead.
- **Confusing sharding with tensor parallelism.** Sharding (FSDP) splits state across data-parallel ranks and reconstructs it for compute. Tensor parallelism splits the *compute* of a single layer; the math is different.
