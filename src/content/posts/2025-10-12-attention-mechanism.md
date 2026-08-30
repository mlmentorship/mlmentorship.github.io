---
title: "The attention mechanism"
description: "Compute a weighted sum of values, weights derived from query-key similarity. The single operation that powers transformers, retrieval, and most of modern ML."
date: "2025-10-12"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

For queries $Q \in \mathbb{R}^{n_q \times d}$, keys $K \in \mathbb{R}^{n_k \times d}$, and values $V \in \mathbb{R}^{n_k \times d_v}$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V.
$$

Each query is replaced by a weighted average of values, with weights given by query-key similarities normalized by softmax.

Attention is a central architectural primitive in modern ML:

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

<!-- visual:attention-keys-choose-values-contribute -->
<figure class="learning-figure" aria-labelledby="attention-routing-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="attention-routing-title">How do keys choose what to retrieve while values supply the result?</p>
	<div class="visual-grid--two" role="group" aria-label="Two-stage single-query attention example">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 220" role="img" aria-labelledby="attention-weights-title attention-weights-desc">
				<title id="attention-weights-title">The query and keys determine three attention weights</title>
				<desc id="attention-weights-desc">One query is compared with three keys. After scaled similarity and softmax, the key rows receive normalized weights 0.60, 0.30, and 0.10. The weights sum to one. Values do not participate in this scoring stage.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="185" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">1 · KEYS SET THE MIXING WEIGHTS</text>
				<rect class="viz-node viz-node--input" x="18" y="91" width="52" height="38" rx="3"></rect>
				<text class="viz-node-label" x="44" y="108">q</text>
				<text class="viz-node-value" x="44" y="122">query</text>
				<rect class="viz-node" x="103" y="38" width="54" height="34" rx="3"></rect>
				<rect class="viz-node" x="103" y="93" width="54" height="34" rx="3"></rect>
				<rect class="viz-node" x="103" y="148" width="54" height="34" rx="3"></rect>
				<text class="viz-callout" x="130" y="59" text-anchor="middle">k₁</text>
				<text class="viz-callout" x="130" y="114" text-anchor="middle">k₂</text>
				<text class="viz-callout" x="130" y="169" text-anchor="middle">k₃</text>
				<path d="M70 103L103 61M70 110H103M70 117L103 159" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<text class="viz-label" x="89" y="88" text-anchor="middle">scaled</text>
				<text class="viz-label" x="89" y="100" text-anchor="middle">similarity</text>
				<rect x="175" y="41" width="48" height="28" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
				<rect x="175" y="96" width="24" height="28" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
				<rect x="175" y="151" width="8" height="28" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
				<path d="M157 55H175M157 110H175M157 165H175" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<text class="viz-callout" x="229" y="59">α₁ = 0.60</text>
				<text class="viz-callout" x="205" y="114">α₂ = 0.30</text>
				<text class="viz-callout" x="189" y="169">α₃ = 0.10</text>
				<text class="viz-axis-label" x="150" y="199" text-anchor="middle">SOFTMAX WEIGHTS: 0.60 + 0.30 + 0.10 = 1</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 220" role="img" aria-labelledby="attention-values-title attention-values-desc">
				<title id="attention-values-title">The same weights mix the three values into the output</title>
				<desc id="attention-values-desc">For a one-dimensional teaching example, values 1, 4, and 7 are multiplied by weights 0.60, 0.30, and 0.10. Their contributions 0.60, 1.20, and 0.70 sum to the attention output 2.50. Keys do not become part of the output.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="185" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">2 · VALUES SUPPLY THE PAYLOAD</text>
				<rect class="viz-node viz-node--input" x="17" y="39" width="166" height="36" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="17" y="92" width="166" height="36" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="17" y="145" width="166" height="36" rx="3"></rect>
				<text class="viz-callout" x="100" y="61" text-anchor="middle">0.60 × v₁ (1) = 0.60</text>
				<text class="viz-callout" x="100" y="114" text-anchor="middle">0.30 × v₂ (4) = 1.20</text>
				<text class="viz-callout" x="100" y="167" text-anchor="middle">0.10 × v₃ (7) = 0.70</text>
				<path d="M183 57L221 94M183 110H221M183 163L221 126" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<rect class="viz-node viz-node--output" x="221" y="84" width="62" height="52" rx="4"></rect>
				<text class="viz-node-label" x="252" y="105">output</text>
				<text class="viz-node-value" x="252" y="123">2.50</text>
				<text class="viz-axis-label" x="17" y="200">WEIGHTED SUM: 0.60 + 1.20 + 0.70</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the query only with the keys to get weights that sum to one. Then carry those same weights across to the paired values and add their contributions. Keys decide the mixture; values are what gets mixed.</figcaption>
</figure>

The $\sqrt{d}$ scaling is critical: without it, the variance of an unscaled dot product grows with $d$ (and its typical magnitude with $\sqrt{d}$), pushing softmax into saturation regions where gradients vanish.

## Self-attention vs. cross-attention

- **Self-attention**: $Q, K, V$ are all derived from the same input. Each token attends to all other tokens in the same sequence. Used in transformer encoder layers and the self-attention sub-block of decoder layers.
- **Cross-attention**: $Q$ comes from one source (e.g., decoder hidden state), $K, V$ from another (e.g., encoder outputs). Used in encoder-decoder transformers (T5, NMT) and modern diffusion text conditioning.

## Multi-head attention

Run attention $H$ times in parallel with different learned $Q, K, V$ projections (each of dimension $d / H$), concatenate the outputs, project back. Each "head" can specialize to different relationships (syntactic, semantic, positional). Standard transformer uses 8–96 heads.

In modern LLMs, heads are reduced via [grouped-query attention](/concepts/grouped-query-attention/) where multiple Q heads share K/V heads.

## Causal (autoregressive) masking

For decoder language models: token $t$ should not attend to tokens $t+1, t+2, \dots$. Implement by adding $-\infty$ to the corresponding entries of $Q K^\top$ before softmax. After softmax, those positions become 0 weight.

This is what enables next-token prediction without leakage.

## Connection to retrieval

Dot-product attention with a single query against many keys is mathematically identical to nearest-neighbor retrieval with cosine similarity (after softmax). The softmax just turns the top-$k$ retrieval into a soft weighting.

## Cost

- Forward: $O(n_q \cdot n_k \cdot d) + O(n_q \cdot n_k \cdot d_v)$ FLOPs.
- Memory: $O(n_q \cdot n_k)$ for the attention matrix. The dominant cost at long context.

[FlashAttention](/concepts/flashattention/) reorders the computation to never materialize the full matrix, dropping memory to $O(n)$.

## Common pitfalls

- **Forgetting the $\sqrt{d}$ scaling.** Softmax saturates; gradients vanish; training fails.
- **Wrong masking for causal LM.** Off-by-one errors leak future tokens; quality looks great in training but inference is broken.
- **Treating attention weights as interpretation.** Attention weights show *what was averaged*, not *what was used*; downstream computation may ignore the weighted result. Don't over-interpret heatmaps.
- **Confusing attention with self-attention.** Attention is the general operation; self-attention is one usage.

## Related

- [Transformer architecture](/concepts/transformer-architecture/). Full assembly.
- [FlashAttention](/concepts/flashattention/). Efficient implementation.
- [Grouped-query attention](/concepts/grouped-query-attention/). Modern KV-cache optimization.
