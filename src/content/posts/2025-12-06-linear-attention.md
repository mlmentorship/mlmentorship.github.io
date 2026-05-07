---
title: "Linear attention (Linformer, Performer, kernel methods)"
description: "Approximate the softmax attention matrix with a low-rank or kernel factorization so cost is linear in sequence length."
date: "2025-12-06"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Linear attention replaces the $n \times n$ softmax matrix with an explicit factorization through a low-dimensional space, so the per-layer cost drops from $O(n^2 d)$ to $O(n k d)$ for some $k \ll n$.

## Why it matters

Sparse attention (see [sparse attention](/reference/sparse-attention/)) keeps the softmax exact but on fewer pairs. Linear attention approximates the softmax itself, exploiting the empirical observation that the $n \times n$ attention matrix is approximately low-rank.

In practice, modern decoder LLMs do not use linear attention. Quality drops are non-trivial at scale and FlashAttention has made dense attention competitive in wall-clock. Linear attention is most relevant in domains with extreme $n$ (genomics, time series of millions of steps) or in research on sub-quadratic alternatives.

## Two main families

### Project the sequence axis (Linformer, [Wang et al., 2020](https://arxiv.org/abs/2006.04768))

Learn fixed projection matrices $E, F \in \mathbb{R}^{k \times n}$ with $k \ll n$. Replace $K, V \in \mathbb{R}^{n \times d}$ with $E K, F V \in \mathbb{R}^{k \times d}$:

$$
\text{Attn} = \text{softmax}\!\left(\frac{Q (EK)^\top}{\sqrt{d}}\right) (FV).
$$

The softmax is now $n \times k$. Cost: $O(n k d)$, linear in $n$. Caveat: $k$ is fixed at training time, so you cannot extrapolate to longer sequences without re-training.

### Replace softmax with a kernel (Performer, [Choromanski et al., 2020](https://arxiv.org/abs/2009.14794))

Softmax can be written as a kernel $K(q, k) = \exp(q^\top k / \sqrt{d})$. Approximate this kernel with random features $\phi: \mathbb{R}^d \to \mathbb{R}^r$ such that $\mathbb{E}[\phi(q)^\top \phi(k)] \approx K(q, k)$.

Then $\text{softmax}(QK^\top) V \approx \phi(Q) (\phi(K)^\top V)$. The right-hand side is computed right-to-left: $\phi(K)^\top V$ is $r \times d$, then $\phi(Q) \cdot (\dots)$ is $n \times d$. Cost: $O(n r d)$, linear in $n$, and works for arbitrary $n$ at inference (no fixed projection).

## When to use linear attention in 2026

- Sequence length $n \gg 32{,}000$ where FlashAttention is still too slow or doesn't fit memory.
- Encoder-only models on very long inputs.
- Real-time inference with strict latency budgets and tolerable quality loss.

For chat-style decoder LLMs, dense attention with FlashAttention + GQA + KV cache remains the production default.

## Common pitfalls

- **Comparing FLOPs without measuring wall-clock.** Linear attention's $O(n)$ scaling only beats $O(n^2)$ FlashAttention at large $n$; the crossover is implementation-dependent and often higher than naive analysis suggests.
- **Forgetting the constants.** Linformer's $k$ and Performer's $r$ may need to be hundreds for good quality, so the linear scaling has a large constant.
- **Assuming all softmax-replacement schemes preserve the autoregressive mask trivially.** Kernelized attention requires careful handling for causal masking (recursive cumulative sums).
