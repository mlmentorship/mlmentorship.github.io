---
title: "FSDP and ZeRO: sharding optimizer state, gradients, and parameters"
description: "How modern training scales beyond a single GPU's memory by partitioning the optimizer state, gradients, and parameters across the data-parallel group."
date: "2025-11-14"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["ZeRO-3", "ZeRO stage 3", "fully sharded data parallel", "parameter sharding"]
---

## Summary

ZeRO stages progressively partition optimizer state, gradients, and model parameters across data-parallel GPUs. Fully Sharded Data Parallel (FSDP) partitions all three persistent state types. Earlier ZeRO stages leave some state replicated.

Training memory has four big consumers:

| Component | Bytes per parameter (BF16 + FP32 master + Adam) |
|-----------|----------------------|
| Parameters (BF16) | 2 |
| Gradients (BF16) | 2 |
| Adam first moment $m$ (FP32) | 4 |
| Adam second moment $v$ (FP32) | 4 |
| Master weights (FP32, optional) | 4 |
| **Total** | **~12–16 bytes/param** |

A 7B-parameter model therefore needs about 84–112 GB of persistent training state before activations and temporary buffers. The lower value omits separate FP32 master weights. The exact value depends on optimizer and framework storage. Ideal full sharding spreads persistent state across $N$ GPUs, but each GPU also needs transient gathered parameters, communication buffers, and activations.

ZeRO ([Rajbhandari et al., 2019](https://arxiv.org/abs/1910.02054), DeepSpeed) and PyTorch FSDP implement this idea. They are the standard for any training run that doesn't fit in a single GPU's memory and doesn't need full tensor or pipeline parallelism.

## The three stages (ZeRO-1/2/3)

### Stage 1: shard optimizer state
Each GPU holds the full parameters and gradients but only $1/N$ of the Adam moments. After the backward pass, each GPU updates its slice and then all-gathers updated parameters.
Memory reduction: up to about 4× for large groups, depending on master-weight storage.

### Stage 2: shard optimizer state + gradients
Same as Stage 1 plus gradients are reduced-scattered (each GPU keeps its slice) instead of all-reduced.
Memory reduction: up to about 8× for large groups under the 16-byte assumption.

### Stage 3 (FSDP): shard optimizer state + gradients + parameters
Each GPU holds only its persistent slice of the parameters. It all-gathers one wrapped unit before computing that unit. If the full parameters are freed after forward, they must be gathered again for backward. Keeping them through backward saves that second gather but raises peak memory.
Memory reduction: nearly $N$× for persistent model state. Peak memory also includes at least one gathered unit and temporary buffers.

PyTorch FSDP and DeepSpeed ZeRO-3 are common implementations of full parameter sharding.

## Tradeoffs

- **Memory vs. communication**: each stage trades more communication for less memory.
- **Sharding granularity**: FSDP can wrap individual layers ("auto-wrap policy") so all-gathers cover only one layer's parameters at a time, capping peak unsharded memory.
- **Mixing with tensor parallelism**: FSDP shards across the data-parallel dimension. Very large runs often combine it with tensor parallelism on the fastest links. The exact placement follows the cluster topology.

## When to use what

| Constraint | Strategy |
|-----------|----------|
| Fits on 1 GPU | DDP (no sharding) |
| Persistent optimizer state is the first limit | ZeRO-1 |
| Optimizer state and gradients are the limit | ZeRO-2 |
| Stored parameters are also the limit | FSDP / ZeRO-3 |
| One gathered layer is too large | Add tensor parallelism |
| Depth or topology needs another split | Consider pipeline parallelism |

## Common pitfalls

- **FSDP alone does not shard activation dimensions.** Each GPU still holds activations for its data-parallel slice. Use [activation checkpointing](/concepts/activation-checkpointing/) or sequence/context parallelism when those activations dominate.
- **All-gather overhead at small layer size.** Wrapping every linear layer separately can dominate runtime; wrap at transformer-block granularity instead.
- **Confusing sharding with tensor parallelism.** Sharding (FSDP) splits state across data-parallel ranks and reconstructs it for compute. Tensor parallelism splits the *compute* of a single layer; the math is different.

## Related

- [Strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/). Choose the lightest sharding stage that fits.
- [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/). Connect layouts with all-gather and reduce-scatter.
- [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/). Build the memory budget first.
