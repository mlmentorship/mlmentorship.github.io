---
title: "Pipeline parallelism"
description: "Split the model across GPUs by layer; pipeline mini-batches through the stages. The way to scale across slow interconnects when TP isn't viable."
date: "2026-04-06"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Pipeline parallelism** (PP) splits a model along its depth: GPU 0 holds layers 1–8, GPU 1 holds layers 9–16, etc. A mini-batch is divided into smaller **micro-batches** that flow through the stages so that GPU 0 starts processing micro-batch 2 while GPU 1 processes micro-batch 1, achieving parallel utilization despite the sequential layer dependency.

## Why it matters

For large models that don't fit on a single GPU, TP is the natural choice **within a node** (where NVLink is fast). But TP doesn't extend across nodes. Communication kills throughput. PP scales across nodes via much smaller cross-stage messages (just the activations between consecutive stages, not weights), enabling **multi-node scaling** of frontier models.

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

Modern frontier training uses interleaved 1F1B or Zero Bubble.

## Cost model

For a model with $L$ layers split across $P$ stages and $M$ micro-batches:

- **Bubble**: $(P - 1) / (P + M - 1)$ of total time. Minimize by increasing $M$.
- **Communication per micro-batch**: send activations of one micro-batch between adjacent stages. Cost $\sim$ activation size $\sim$ batch × seq × hidden, much smaller than full weight all-reduce.
- **Activation memory per stage**: in 1F1B, $\sim P$ micro-batches' worth of activations.

## When PP wins

- **Cross-node scaling** with slow interconnect.
- **Very deep models** where one stage easily fits on a node.
- **Frontier training** combining 3D parallelism (DP + TP + PP).

## When PP loses

- **Small models** that fit on one node. TP within node + DP across nodes is simpler.
- **Few micro-batches** in a step. Bubble dominates.
- **Workloads with very different per-layer compute**. Load imbalance creates idle GPUs.

## 3D parallelism

The standard frontier training stack:

- **Tensor parallel** within a node (4–8 GPUs, NVLink).
- **Pipeline parallel** across small groups of nodes.
- **Data parallel / FSDP** across remaining nodes (sharded for memory).

For a 405B-parameter model on 1024 GPUs: TP=8, PP=16, DP=8 is a typical configuration.

## Common pitfalls

- **Few micro-batches → big bubble.** Use $M \ge 4P$ as a rule of thumb.
- **Imbalanced stage compute.** Layer 1 (embedding) and the last layer (LM head) may be much heavier or lighter than middle layers. Manual partitioning helps.
- **Forgetting activation memory grows with $M$ in 1F1B.** Combine with activation checkpointing.
- **Treating PP as the same as TP.** Different sharding axes, different communication patterns. PP is bandwidth-light; TP is bandwidth-heavy.
- **Skipping interleaving on >4 stages.** Sequential 1F1B has noticeable bubble at large $P$; interleaving cuts it.

## Related

- [Tensor parallelism](/concepts/tensor-parallelism/). Orthogonal sharding within nodes.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Orthogonal sharding for memory.
- [Activation checkpointing](/concepts/activation-checkpointing/). Reduce per-stage activation memory.
