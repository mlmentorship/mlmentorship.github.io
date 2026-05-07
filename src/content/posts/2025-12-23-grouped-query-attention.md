---
title: "Grouped-query and multi-query attention (GQA, MQA)"
description: "Share K and V heads across query heads to shrink the KV cache 4-8x with negligible quality loss. Standard in modern decoder LLMs."
date: "2025-12-23"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

GQA and MQA reduce the number of distinct K/V projection heads while keeping the full set of Q heads, so multiple query heads share the same key and value tensors. MQA is the extreme case: one K/V head total. GQA picks an intermediate number of K/V groups.

## Why it matters

The KV cache dominates LLM serving memory at long contexts (see [KV cache](/reference/kv-cache/)). Cutting the number of K/V heads cuts cache size proportionally:

- **MHA** (standard, e.g. GPT-3): K and V have the same number of heads as Q.
- **MQA** [(Shazeer, 2019)](https://arxiv.org/abs/1911.02150): 1 K and 1 V head shared across all Q heads. ~`num_heads`× smaller cache.
- **GQA** [(Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245): G groups, each shared across `num_heads / G` Q heads. Tunable midpoint.

Llama 2 70B uses GQA with 8 K/V groups for 64 query heads (8× cache reduction). Llama 3, Mistral, Qwen, and most modern decoders default to GQA.

## The mechanism

In standard multi-head attention, for each head $h$:
$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d}}\right) V_h
$$
with $Q_h, K_h, V_h \in \mathbb{R}^{n \times d}$ and $h \in \{1, \dots, H\}$.

In GQA with $G$ groups, the $H$ query heads are partitioned into $G$ contiguous groups of size $H/G$. All query heads in the same group attend to the same shared $K_g, V_g$. MQA is GQA with $G = 1$.

Implementation: project K and V to dimension $G \cdot d$ instead of $H \cdot d$, then broadcast (repeat) across the matching Q heads before the matmul.

## Tradeoffs

| Variant | KV heads | Cache size | Quality | Used by |
|---------|---------|-----------|---------|---------|
| MHA | $H$ | 1× | baseline | GPT-3, original Llama |
| GQA-8 | 8 | $H/8$× | ~baseline | Llama 2/3 70B, Mistral |
| MQA | 1 | $1/H$× | small drop | PaLM, Falcon |

GQA recovers nearly all MHA quality while keeping most of MQA's cache savings. The dominant choice in 2026.

## Common pitfalls

- **Confusing K/V heads with Q heads.** GQA shrinks K/V only; Q stays full-rank.
- **Assuming the speedup is in compute.** GQA mostly saves *memory* (cache + bandwidth), not FLOPs. The matmul cost barely changes.
- **Re-training cost.** You generally cannot convert MHA → GQA post-hoc; the K/V projections were trained per-head. Distillation or partial re-training is required.
