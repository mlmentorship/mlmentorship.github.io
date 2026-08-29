---
title: "Context parallelism and ring attention"
description: "Shard a long sequence across devices while preserving exact attention and controlling activation memory."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Context parallelism splits sequence positions across devices. Each device stores and computes a local part of the sequence, then exchanges attention data so local queries can attend to all allowed keys and values.

## Why AI labs care

Long-context training can run out of memory even when model weights fit. The main pressure comes from activations and attention state that grow with sequence length.

Context parallelism can reduce memory per device without changing the model's attention rule. It is different from sparse or approximate attention.

## Sequence parallelism is not the same thing

The names are sometimes used differently across libraries. A useful distinction is:

- **Sequence parallelism** shards sequence positions for operations outside attention, such as normalization, dropout, and residual work. Tensor-parallel layers often already require layout changes that make this cheap.
- **Context parallelism** shards the sequence inside attention itself. It requires communication because each query may need keys and values from other devices.

Ask which definition a system uses before comparing results.

## The attention problem

For one head:

$$
O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{H}} + M\right)V,
$$

where $M$ contains causal or other attention masks.

Split a sequence of length $T$ across $R$ devices. Each device holds about $T/R$ local query, key, and value positions.

Local queries still need remote keys and values. A local-only attention calculation would change the model.

## Ring attention

Ring attention keeps each device's query block local and sends key/value blocks around a ring.

At each step:

1. compute attention between local queries and the current key/value block;
2. update a running softmax result;
3. send that key/value block to the next device;
4. receive another block;
5. repeat until every local query has seen every allowed key/value block.

The output is exact apart from normal floating-point effects.

## Online softmax

A device cannot normalize after each block independently. The final softmax denominator includes scores from all blocks.

For each query row, keep:

- the largest score seen so far;
- a running sum of exponentials adjusted to that maximum;
- a running weighted value sum.

Let $m$, $z$, and $u$ be those three running values. Let $m_b$, $z_b$, and $u_b$ be the values for a new block. Update them with:

$$
m' = \max(m,m_b),
$$

$$
z' = e^{m-m'}z + e^{m_b-m'}z_b,
\qquad
u' = e^{m-m'}u + e^{m_b-m'}u_b.
$$

After all blocks, the output is $u'/z'$. Rescaling prevents overflow and makes the result equal to one softmax over all visible blocks, apart from floating-point rounding.

## Causal attention

A token may attend only to its current or earlier positions. Devices need global position ranges for each query and key block.

A device can:

- skip a block that is entirely in the future;
- process a block that is entirely valid;
- apply a triangular mask when query and key ranges overlap.

Good block order can avoid work on future blocks.

## Memory and compute

With even sharding:

- local sequence activations fall by about $R$;
- local query positions fall by about $R$;
- total exact-attention work across the group is still based on the full sequence;
- each device communicates key/value blocks around the group.

The method helps memory and divides compute. It does not remove the quadratic total work of exact attention.

## Communication choices

A ring uses point-to-point transfers and can pipeline communication with attention compute.

Some systems use all-gather or all-to-all layouts instead. The best choice depends on:

- physical network topology;
- number of devices;
- local sequence length;
- number of key/value heads;
- whether tensor parallelism is also active;
- which layout the next operation needs.

Grouped-query or multi-query attention reduces key/value bytes and therefore reduces context-parallel communication.

## Load balance

Simple causal sharding can create uneven work. Early query blocks have fewer valid key blocks than late query blocks.

Balanced position assignment can pair early and late sequence pieces on each device. This improves utilization while preserving token order in the attention mask.

Padding and variable sequence lengths can create another imbalance. Packing and length-aware assignment help.

## When to use it

Use context parallelism when:

- activation memory is dominated by long sequences;
- reducing the local micro-batch is not enough;
- tensor or fully sharded parallelism does not split the needed sequence state;
- the network can carry key/value traffic efficiently.

Do not add it for short sequences that already fit. The communication and implementation complexity may reduce speed.

## In an interview

Use this order:

1. Separate model-state memory from sequence activation memory.
2. State the global and per-device sequence lengths.
3. Explain why local queries need remote keys and values.
4. Describe ring transfer and stable online softmax.
5. Handle causal masks and global positions.
6. Estimate key/value bytes per block.
7. Discuss overlap, topology, and load balance.
8. Compare with sparse attention only if changing model behavior is allowed.

## Common mistakes

- Calling context parallelism sparse attention.
- Computing a separate softmax for each remote block.
- Ignoring causal masking across device boundaries.
- Claiming the method removes quadratic total compute.
- Forgetting key/value communication.
- Combining tensor and context groups without checking the physical topology.
- Using one library's naming as a universal definition.

*Related: [FlashAttention](/concepts/flashattention/), [long-context LLMs](/concepts/long-context-llms/), [tensor parallelism](/concepts/tensor-parallelism/), and [accelerator network topology](/concepts/accelerator-network-topology/). Further reading: [training parallelism in the JAX Scaling Book](https://jax-ml.github.io/scaling-book/training).*