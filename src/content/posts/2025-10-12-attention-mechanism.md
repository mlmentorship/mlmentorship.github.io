---
title: "The attention mechanism"
description: "Compute a weighted sum of values, weights derived from query-key similarity. The single operation that powers transformers, retrieval, and most of modern ML."
date: "2025-10-12"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

For queries $Q \in \mathbb{R}^{n_q \times d}$, keys $K \in \mathbb{R}^{n_k \times d}$, and values $V \in \mathbb{R}^{n_k \times d_v}$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V.
$$

Each query is replaced by a weighted average of values, with weights given by query-key similarities normalized by softmax.

## Why it matters

Attention is the single most important architectural primitive of the past decade:

- **Transformers** are stacks of attention + FFN. Every modern LLM is mostly attention by parameter and FLOP count.
- **Retrieval** (two-tower, cross-encoder) is dot-product attention between queries and items.
- **Vision transformers**, **graph attention networks**, **diffusion models** all use it.
- **Memory-augmented networks** use attention to access external memory.

Understanding attention at the computational and conceptual level is non-negotiable for senior ML.

## The mechanism step by step

For a single query $q \in \mathbb{R}^d$ and set of $n$ key-value pairs:

1. **Score** each key against the query: $s_i = q^\top k_i / \sqrt{d}$.
2. **Normalize** with softmax: $\alpha_i = \exp(s_i) / \sum_j \exp(s_j)$. The $\alpha_i$ sum to 1. They form an attention distribution.
3. **Aggregate**: output $= \sum_i \alpha_i v_i$.

Each output is a convex combination of values, biased toward keys most similar to the query.

The $\sqrt{d}$ scaling is critical: without it, dot products grow linearly in $d$, pushing softmax into saturation regions where gradients vanish.

## Self-attention vs. cross-attention

- **Self-attention**: $Q, K, V$ are all derived from the same input. Each token attends to all other tokens in the same sequence. Used in transformer encoder layers and the self-attention sub-block of decoder layers.
- **Cross-attention**: $Q$ comes from one source (e.g., decoder hidden state), $K, V$ from another (e.g., encoder outputs). Used in encoder-decoder transformers (T5, NMT) and modern diffusion text conditioning.

## Multi-head attention

Run attention $H$ times in parallel with different learned $Q, K, V$ projections (each of dimension $d / H$), concatenate the outputs, project back. Each "head" can specialize to different relationships (syntactic, semantic, positional). Standard transformer uses 8–96 heads.

In modern LLMs, heads are reduced via [grouped-query attention](/reference/grouped-query-attention/) where multiple Q heads share K/V heads.

## Causal (autoregressive) masking

For decoder language models: token $t$ should not attend to tokens $t+1, t+2, \dots$. Implement by adding $-\infty$ to the corresponding entries of $Q K^\top$ before softmax. After softmax, those positions become 0 weight.

This is what enables next-token prediction without leakage.

## Connection to retrieval

Dot-product attention with a single query against many keys is mathematically identical to nearest-neighbor retrieval with cosine similarity (after softmax). The softmax just turns the top-$k$ retrieval into a soft weighting.

## Cost

- Forward: $O(n_q \cdot n_k \cdot d) + O(n_q \cdot n_k \cdot d_v)$ FLOPs.
- Memory: $O(n_q \cdot n_k)$ for the attention matrix. The dominant cost at long context.

[FlashAttention](/reference/flashattention/) reorders the computation to never materialize the full matrix, dropping memory to $O(n)$.

## Common pitfalls

- **Forgetting the $\sqrt{d}$ scaling.** Softmax saturates; gradients vanish; training fails.
- **Wrong masking for causal LM.** Off-by-one errors leak future tokens; quality looks great in training but inference is broken.
- **Treating attention weights as interpretation.** Attention weights show *what was averaged*, not *what was used*; downstream computation may ignore the weighted result. Don't over-interpret heatmaps.
- **Confusing attention with self-attention.** Attention is the general operation; self-attention is one usage.

## Related

- [Transformer architecture](/reference/transformer-architecture/). Full assembly.
- [FlashAttention](/reference/flashattention/). Efficient implementation.
- [Grouped-query attention](/reference/grouped-query-attention/). Modern KV-cache optimization.
