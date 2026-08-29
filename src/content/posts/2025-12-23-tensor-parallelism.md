---
title: "Tensor parallelism"
description: "Split a single matrix multiplication across multiple GPUs. The way to fit one transformer layer that doesn't fit on a single device."
date: "2025-12-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Tensor parallelism** (TP) splits the *computation* of a single layer (typically a matmul) across multiple GPUs by sharding the weight matrix along one of its dimensions. Each GPU computes its slice, and an all-reduce or all-gather aggregates the result before the next layer.

For very large models (70B+, 405B, MoE-1T), a single transformer layer's weights and activations don't fit on a single GPU even with FSDP. TP shards individual layers. Required for frontier-scale training and inference. Combined with pipeline parallelism and FSDP, it forms **3D parallelism** used by modern training stacks.

## How a transformer layer is sharded

The standard sharding from Megatron-LM [(Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053):

### FFN (two matmuls + activation)

```
y = GeLU(x @ W_1) @ W_2
```

- $W_1$ split column-wise: each GPU holds $W_1[:, c_i:c_{i+1}]$. Produces a partial output for its slice of channels. No communication needed up to the GeLU (elementwise).
- $W_2$ split row-wise: each GPU holds $W_2[r_i:r_{i+1}, :]$. Multiplies its slice. Output is summed across GPUs via all-reduce.

Two matmuls with one all-reduce per FFN block.

### Attention

Split heads across GPUs: each GPU computes its subset of attention heads. Output projection $W_O$ is split row-wise, requiring an all-reduce at the end.

Two matmuls (heads, output projection) with one all-reduce per attention block.

## Communication cost

In the common Megatron layout, a forward pass uses one activation reduction after the attention output projection and one after the second feed-forward projection. The backward pass has matching communication for the input gradients of the column-parallel projections.

The message size follows the activation shape, not the parameter count. If an activation tensor contains $A$ bytes and the tensor-parallel degree is $N$, a ring all-reduce moves about:

$$
2A\frac{N-1}{N}
$$

bytes per rank. Sequence-parallel implementations often replace an all-reduce with a reduce-scatter and a later all-gather so the intermediate activation stays split. The exact collective count depends on the layout and framework.

TP communicates every layer, so it needs high effective bandwidth and low latency. It is commonly kept inside a fast accelerator domain. It can cross nodes when the network, message sizes, and local batch provide enough communication efficiency. The decision should come from a cost estimate and a scaling trace, not a fixed node boundary.

## Sequence parallelism

A complement to TP that shards the **sequence dimension** for operations not parallelized by TP, such as LayerNorm, dropout, and residual work. It can reduce those activation tensors by about the tensor-parallel degree. It is not free: it usually changes all-reduce operations into paired reduce-scatter and all-gather operations. This can keep similar communication volume while reducing peak memory.

## TP vs. data parallelism vs. pipeline parallelism

| Sharding axis | Memory savings | Communication |
|---------------|---------------|---------------|
| **DDP / FSDP** (data) | Each GPU sees a different mini-batch | Gradient all-reduce / all-gather |
| **TP** (tensor) | Each GPU shards layer weights and activations | Per-layer all-reduce |
| **PP** (pipeline) | Each GPU holds different layers | Activation send between adjacent stages |
| **Sequence** (within TP) | Reduces activation memory in TP | Reduce-scatter and all-gather layout changes |

**3D parallelism**: combine DP + TP + PP for very large models. Typical config: TP within a node, PP across small groups of nodes, DP across remaining nodes.

## When to use TP

- **Layer too large to fit on single GPU**: even with FSDP all-gather, the unsharded layer must fit. TP keeps the layer sharded throughout.
- **Inference**: TP is a common way to serve models that need several accelerators; major serving runtimes support it.
- **Throughput optimization within a node**: TP with NVLink can be faster than data parallelism for small batch sizes.

## When NOT to use TP

- **When the required links are too slow**: estimate exposed collective time for the real message sizes before extending the group.
- **Small models that fit on one GPU**: pure DP / FSDP is simpler.
- **Pipeline-friendly architectures**: PP can be cheaper communication-wise across slow interconnects.

## Common pitfalls

- **Using TP across a slow interconnect.** Frequent activation collectives can dominate. Keep the group on the fastest useful links unless measurements support a wider group.
- **Assuming TP solves every memory limit.** TP shards layers along selected axes. Add optimizer or full-state sharding only when the remaining state requires it.
- **Sharding embedding tables incorrectly.** The vocab embedding is large ($V \times d$); shard it carefully (Megatron has its own embedding sharding).
- **Communication count math.** Each TP block adds all-reduces; for narrow models / small batches, communication can dominate compute.
- **Tooling ambiguity.** "Tensor parallel size = 8" with mismatched DP / PP can give surprising aggregate batch sizes.

## Related

- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Orthogonal sharding.
- [Pipeline parallelism](/concepts/pipeline-parallelism/). Alternative cross-device split.
- [All-reduce and collectives](/concepts/all-reduce-and-collectives/). The underlying primitives.
- [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/). Derive collectives from tensor layouts.
- [Accelerator network topology](/concepts/accelerator-network-topology/). Place the tensor-parallel group.
