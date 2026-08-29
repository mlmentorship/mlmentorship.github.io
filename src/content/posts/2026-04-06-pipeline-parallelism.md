---
title: "Pipeline parallelism"
description: "Split the model across GPUs by layer; pipeline mini-batches through the stages. The way to scale across slow interconnects when TP isn't viable."
date: "2026-04-06"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Pipeline parallelism** (PP) splits a model along its depth: GPU 0 holds layers 1–8, GPU 1 holds layers 9–16, etc. A mini-batch is divided into smaller **micro-batches** that flow through the stages so that GPU 0 starts processing micro-batch 2 while GPU 1 processes micro-batch 1, achieving parallel utilization despite the sequential layer dependency.

For large models that do not fit on one accelerator, tensor parallelism is often placed on the fastest links because it communicates every layer. Pipeline parallelism sends activations only between adjacent stages, so it can often use slower links more efficiently. Tensor parallelism can cross nodes when the network and workload support it. The choice follows measured communication cost, not a fixed node boundary.

## The basic idea

Without micro-batches, naive pipeline:

```
GPU 0: forward layer 1-8 ────────────  →  backward layer 1-8 ──── 
GPU 1: ───────────  forward 9-16 ────  →  backward 9-16 ───────
GPU 2: ─────────────────  forward 17-24 → backward 17-24 ─
```

Most GPUs are idle most of the time. The **pipeline bubble**.

With micro-batches, the bubble shrinks:

```
GPU 0: f1  f2  f3  f4 ─────────────────────  b4  b3  b2  b1
GPU 1: ─── f1  f2  f3  f4 ─────────  b4  b3  b2  b1 ───────
GPU 2: ─────── f1  f2  f3  f4  b4  b3  b2  b1 ─────────────
```

Bubble fraction $\approx (\text{stages} - 1) / (\text{stages} + \text{micro-batches} - 1)$.

## GPipe vs. 1F1B vs. interleaved

- **GPipe** [(Huang et al., 2018)](https://arxiv.org/abs/1811.06965): all forwards then all backwards. Bubble fraction high; activation memory high (must store all forwards).
- **1F1B** (one forward, one backward; PipeDream): start backward as soon as the first micro-batch reaches the last stage. Reduces bubble and activation memory.
- **Interleaved 1F1B** (Megatron): each GPU holds non-contiguous chunks of layers (e.g., layers 1-2 and 9-10) so the bubble shrinks further.
- **Zero Bubble Pipeline** (recent): split backward into weight-grad and input-grad parts to fill almost all bubbles.

Large training systems may use interleaved 1F1B or zero-bubble schedules when simple schedules leave too much idle time.

## Cost model

For a model with $L$ layers split across $P$ stages and $M$ micro-batches:

- **Bubble**: $(P - 1) / (P + M - 1)$ of total time. Minimize by increasing $M$.
- **Communication per micro-batch**: send activations between adjacent stages. Cost scales with micro-batch × sequence × hidden size. Whether this is cheaper than another layout depends on message frequency, dtype, topology, and overlap.
- **Activation memory per stage**: in 1F1B, $\sim P$ micro-batches' worth of activations.

## When PP wins

- **Cross-node scaling** with slow interconnect.
- **Very deep models** where one stage easily fits on a node.
- **Frontier training** combining 3D parallelism (DP + TP + PP).

## When PP loses

- **Small models** that fit on one node. TP within node + DP across nodes is simpler.
- **Few micro-batches** in a step. The bubble dominates.
- **Workloads with very different per-layer compute**. Load imbalance creates idle GPUs.

## 3D parallelism

One common large-model layout combines:

- **Tensor parallelism** on the fastest useful links.
- **Pipeline parallelism** across balanced groups of layers.
- **Data parallelism or FSDP** across the remaining replica dimension.

The degrees must multiply to the available device count, but that arithmetic does not select the layout. Memory, global batch, topology, and measured scaling select it.

## Common pitfalls

- **Few micro-batches create a large bubble.** More micro-batches improve utilization, but they interact with activation memory and the global batch.
- **Imbalanced stages.** Embeddings, output layers, and uneven blocks can make one stage slower than the others.
- **Forgetting activation memory.** The schedule controls how many micro-batches' activations remain live. Combine PP with activation checkpointing when needed.
- **Treating PP as the same as TP.** They split different axes and communicate at different points. Compare their actual activation messages and schedule overhead.
- **Using a fixed topology rule.** A node boundary is not a law. Measure the links and collectives on the target cluster.
- **Skipping schedule comparison at many stages.** Interleaving can reduce bubbles, but it changes communication and model placement. Measure both schedules.

## Related

- [Tensor parallelism](/concepts/tensor-parallelism/). Orthogonal sharding within a layer.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Orthogonal sharding for memory.
- [Activation checkpointing](/concepts/activation-checkpointing/). Reduce per-stage activation memory.
- [Strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/). Decide whether another pipeline stage helps.
- [Accelerator network topology](/concepts/accelerator-network-topology/). Place stage boundaries on the target cluster.
