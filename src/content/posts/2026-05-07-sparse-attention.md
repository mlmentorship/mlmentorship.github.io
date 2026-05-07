---
title: "Sparse attention (BigBird, Longformer)"
description: "Replace the dense n×n attention mask with a sparse pattern that has O(n) non-zeros while preserving information flow across the full sequence."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Sparse attention is exact attention restricted to a structured sparse mask, so cost scales as $O(n \cdot k)$ for some constant $k \ll n$ instead of $O(n^2)$, while the mask is designed so information can still propagate to all positions in a small number of layers.

## Why it matters

Dense self-attention costs $O(n^2 \cdot d)$ in compute and $O(n^2)$ in memory. For long inputs (long documents, long-form audio, code repos) this becomes the binding constraint. Empirically most attention weights are near zero. So the dense matrix is wasteful.

Sparse attention was the dominant pre-2023 approach to long context. It is still relevant for encoder-only long-document models (clinical notes, legal contracts, scientific papers).

## The two canonical patterns

### Longformer [(Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
- **Sliding window**: each token attends to its $w$ neighbors on each side. Local context, like a 1D CNN receptive field.
- **Global attention**: a small set of designated tokens (e.g., `[CLS]`, question tokens in QA) attend to everything and are attended by everything.

Cost: $O(n \cdot (w + g))$ per layer.

### BigBird [(Zaheer et al., 2020)](https://arxiv.org/abs/2007.14062)
Adds a third component:
- **Window** (local context).
- **Random**: each token attends to $r$ random positions across the sequence (small-world shortcut so any two tokens are connected within $O(\log n)$ hops).
- **Global**: same as Longformer.

Cost: $O(n \cdot (w + r + g)) = O(n)$ for fixed $w, r, g$. BigBird proves that this pattern is a universal approximator of sequence functions and is Turing complete.

## Tradeoffs vs. dense attention

- Quality on standard NLP benchmarks: comparable to dense attention at $n \le 4096$; clearly better at $n \ge 8192$ where dense doesn't fit.
- Wall-clock speedup is implementation-bound. Sparse masks need custom CUDA kernels (or BlockSparse / FlashAttention sparse mode) to actually beat dense FlashAttention.
- Generation quality of decoder-only sparse-attention LLMs has lagged dense-attention LLMs in practice; dense + GQA + KV cache is the standard for chat models. Sparse attention is more common in encoders and retrieval models.

## Common pitfalls

- **Assuming sparsity automatically means speed.** Without a kernel that exploits the structure, you'll be slower than dense FlashAttention.
- **Confusing sparse with low-rank.** Sparse keeps the softmax exact but on fewer pairs; low-rank methods (Linformer, Performer. See [linear attention](/reference/linear-attention/)) approximate the softmax matrix itself.
- **Picking $w$ too small.** With window 32 and 24 layers, information at position 0 cannot reach position 5000 in one forward pass. Either widen the window or add global tokens.
