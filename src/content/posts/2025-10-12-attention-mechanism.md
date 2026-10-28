---
title: "The attention mechanism"
description: "Compute a weighted sum of values, weights derived from query-key similarity. The single operation that powers transformers, retrieval, and most of modern ML."
date: "2025-10-12"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Attention forms one permitted weighted value mixture for each query.
Queries $Q\in\mathbb R^{n_q\times d}$ and keys $K\in\mathbb R^{n_k\times d}$ determine weights.
Values $V\in\mathbb R^{n_k\times d_v}$ supply the mixed payload; key and value rows must remain paired.

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V.
$$

Normalize each query's allowed scores separately, then multiply the resulting $n_q\times n_k$ matrix by $V$.
Without dropout or later projections, every nonempty row is a convex value combination.
This primitive appears in sequence, vision, graph, and memory models; an architecture's dominant cost depends on its actual dimensions.

## The mechanism step by step

For a single query $q \in \mathbb{R}^d$ and set of $n$ key-value pairs:

1. **Score** each key against the query: $s_i = q^\top k_i / \sqrt{d}$.
2. **Normalize** with softmax: $\alpha_i = \exp(s_i) / \sum_j \exp(s_j)$. The $\alpha_i$ sum to 1. They form an attention distribution.
3. **Aggregate**: output $= \sum_i \alpha_i v_i$.

Take one-dimensional query $q=1$, keys $(\log6,\log3,0)$, and values $(1,4,7)$.
Exponentiated scores are $(6,3,1)$, so weights are $(0.6,0.3,0.1)$.
The output is $0.6(1)+0.3(4)+0.1(7)=2.5$.
Changing a value changes the mixture's payload without changing these query-key scores.

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
	<figcaption><strong>Read it this way:</strong> query-key scores determine weights summing to one. Those same weights mix the paired values into 2.5. Keys select the mixture; values supply its contents. <span class="visual-reader-check" data-reader-check="attention-keys-choose-values-contribute"><strong>Check:</strong> Why does the value with weight 0.30 contribute more than the value with weight 0.60?</span></figcaption>
</figure>

If corresponding query-key coordinates are independent, zero-mean, and unit-variance, their summed dot product has variance $d$.
Dividing by $\sqrt d$ gives unit variance under those assumptions.
Learned correlations or different coordinate scales can change that analysis; scaling is not a universal nonsaturation guarantee.

## Self-attention vs. cross-attention

- **Self-attention:** queries, keys, and values derive from the same sequence; a mask may restrict visible positions.
- **Cross-attention:** queries derive from one source and paired keys/values from another; their sequence lengths can differ.

## Multi-head attention

Apply learned projections for several heads, mix values separately, concatenate the outputs, and apply an output projection.
Head dimensions and counts are architecture choices; semantic specialization must be demonstrated rather than assumed.

In modern LLMs, heads are reduced via [grouped-query attention](/concepts/grouped-query-attention/) where multiple Q heads share K/V heads.

## Causal (autoregressive) masking

**New task:** Exclude future or padded positions from each mixture.

**Picture and change:** Set forbidden scores to $-\infty$ before normalization; keep absolute position and cache-offset conventions explicit.

**Check:** Excluding the third key in the example renormalizes weights to $(2/3,1/3,0)$ and changes output to $2$.
Simply zeroing its old weight without renormalization gives $1.8$, a different operation.

**Use when:** Only a permitted subset may contribute.
An entirely masked row has no categorical distribution; reject it or define a separate zero-output policy explicitly.

## Connection to retrieval

**New task:** Return one retrieved item instead of a value mixture.

**Picture and change:** Replace aggregation with the maximum-score key's associated item.

**Check:** The example selects the first key and value $1$, not the attention output $2.5$.

**Use when:** A hard candidate is required.
Positive-temperature softmax preserves score order, but a weighted sum is not nearest-neighbor retrieval.
Cosine and raw dot-product rankings agree only under appropriate normalization, not in general.

## Cost

- Score and mixture arithmetic costs $O(n_qn_k(d+d_v))$, excluding projections.
- Explicit scores and weights need $O(n_qn_k)$ intermediate storage, in addition to inputs and $O(n_qd_v)$ output.

[FlashAttention](/concepts/flashattention/) tiles exact dense attention without retaining the full score matrix.
Its storage still includes dimension-dependent inputs, outputs, row statistics, and tile workspace.

<!-- learning:core-end -->
<div data-learning-extension>

## Common pitfalls

- **Unpaired rows.** Permuting keys without the same value permutation changes the operation.
- **Wrong causal offset.** Cached decoding compares absolute query and key positions, not an arbitrary local triangle.
- **Explanatory overreach.** Mixture weights do not establish a causal explanation for the final model prediction.
- **All-masked rows.** Subtracting a row maximum of negative infinity produces invalid arithmetic without a special policy.

## Reconstruct

<details>
<summary>What survives a joint permutation of keys and values?</summary>

Permuting paired keys and values merely permutes weights and their corresponding payloads, leaving the final sum unchanged.
The mask and position-dependent score terms must receive the same permutation.
Permuting only keys breaks that pairing guarantee.

</details>

## Related

- [Transformer architecture](/concepts/transformer-architecture/). Full assembly.
- [FlashAttention](/concepts/flashattention/). Efficient implementation.
- [Grouped-query attention](/concepts/grouped-query-attention/). Modern KV-cache optimization.

</div>
