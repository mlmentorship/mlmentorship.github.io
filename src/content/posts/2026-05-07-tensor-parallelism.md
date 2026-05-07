---
title: "Tensor parallelism"
description: "Split a single matrix multiplication across multiple GPUs. The way to fit one transformer layer that doesn't fit on a single device."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

**Tensor parallelism** (TP) splits the *computation* of a single layer (typically a matmul) across multiple GPUs by sharding the weight matrix along one of its dimensions. Each GPU computes its slice, and an all-reduce or all-gather aggregates the result before the next layer.

## Why it matters

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

Per forward pass, TP requires 2 all-reduces per transformer block (one per FFN, one per attention). All-reduce cost scales as $O(N \cdot \text{params}/N)$. Same as without sharding, but split into many small messages. Bandwidth-bound.

For backward pass: same number of all-reduces. So TP doubles the communication compared to a single-GPU forward.

**Critical**: TP communication must use high-bandwidth interconnect (NVLink within a server, NVSwitch). Across PCIe or InfiniBand, TP throughput collapses. **TP is typically restricted to within a single node** (4–8 GPUs).

## Sequence parallelism

A complement to TP that shards the **sequence dimension** for operations not parallelized by TP (LayerNorm, dropout, residual). Saves activation memory by ~$N$× without extra all-reduce overhead. Used in NVIDIA's Megatron and most modern stacks.

## TP vs. data parallelism vs. pipeline parallelism

| Sharding axis | Memory savings | Communication |
|---------------|---------------|---------------|
| **DDP / FSDP** (data) | Each GPU sees a different mini-batch | Gradient all-reduce / all-gather |
| **TP** (tensor) | Each GPU shards layer weights and activations | Per-layer all-reduce |
| **PP** (pipeline) | Each GPU holds different layers | Activation send between adjacent stages |
| **Sequence** (within TP) | Reduces activation memory in TP | Free; done with TP |

**3D parallelism**: combine DP + TP + PP for very large models. Typical config: TP within a node, PP across small groups of nodes, DP across remaining nodes.

## When to use TP

- **Layer too large to fit on single GPU**: even with FSDP all-gather, the unsharded layer must fit. TP keeps the layer sharded throughout.
- **Inference**: TP is the standard way to serve large models; vLLM, TGI, TensorRT-LLM all support TP.
- **Throughput optimization within a node**: TP with NVLink can be faster than data parallelism for small batch sizes.

## When NOT to use TP

- **Across PCIe / Ethernet**: communication overhead dominates.
- **Small models that fit on one GPU**: pure DP / FSDP is simpler.
- **Pipeline-friendly architectures**: PP can be cheaper communication-wise across slow interconnects.

## Common pitfalls

- **Using TP across slow interconnect.** Bandwidth-limited; use only with NVLink / NVSwitch (within a single node typically).
- **Forgetting to combine with FSDP.** TP shards layers, FSDP shards optimizer/grads/params; both can run together.
- **Sharding embedding tables incorrectly.** The vocab embedding is large ($V \times d$); shard it carefully (Megatron has its own embedding sharding).
- **Communication count math.** Each TP block adds all-reduces; for narrow models / small batches, communication can dominate compute.
- **Tooling ambiguity.** "Tensor parallel size = 8" with mismatched DP / PP can give surprising aggregate batch sizes.

## Related

- [FSDP and ZeRO](/reference/fsdp-and-zero/). Orthogonal sharding.
- [Pipeline parallelism](/reference/pipeline-parallelism/). Alternative cross-device split.
- [All-reduce and collectives](/reference/all-reduce-and-collectives/). The underlying primitives.
