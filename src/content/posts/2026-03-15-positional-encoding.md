---
title: "Positional encoding: where does position enter attention?"
description: "Compare embeddings, rotary geometry, and score biases; compute ALiBi probabilities and test slope, visibility, and common-shift changes."
date: "2026-03-15"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Position can change embeddings, query/key geometry, or scores before softmax.
Those are different computations, not interchangeable ways to relabel a checkpoint.
Unmasked position-free self-attention is permutation-equivariant: permuting input rows permutes output rows.
A fixed causal mask already restricts positions asymmetrically.

**Task:** query position $i=6$ may read keys $j=[2,4,6]$.
Hold scaled content scores at zero and use causal ALiBi slope $s_h=0.25$.
Compute the attention probabilities, then change the slope and permitted directions.

## The lineup

The pipeline is embeddings, Q/K projections, dot product, bias/mask, then softmax.
Locate each method at its actual stage.

### Sinusoidal (original Transformer, 2017)

Add fixed absolute-position vectors before the projections:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Defined values at unseen positions do not guarantee useful extrapolation.

### Learned absolute positional embeddings

Add a learned table row for each supported position, as in BERT or GPT-2.
Unseen indices need an explicit adaptation, not an undefined table lookup.

### Relative positional encoding

T5-style relative bucket biases enter scores; bucket definitions are part of the model.
Shaw and Transformer-XL use other relative constructions, not the same lookup formula.

<span id="rope-rotary-position-embeddings-the-modern-default"></span>
### RoPE (Rotary Position Embeddings)

Rotate query/key coordinate pairs before their dot product, rather than adding an embedding vector.

For positions `m` and `n`, use their specified block-diagonal rotations:
```
Q' = R_m * Q   (Q rotated by query position)
K' = R_n * K   (K rotated by key position)
```

The [rotation derivation](/concepts/rotary-position-embeddings/) proves the common-shift relationship for fixed content vectors.
The figure retains positions $(1,3)$ and $(4,6)$; their relative offsets agree.

### ALiBi (Attention with Linear Biases)

Add a distance bias after the scaled content score:
```
score_h(i, j) = Q_i K_j^T / sqrt(d) - s_h * (i - j)
```
Here $d$ is head width, $s_h>0$ is a fixed head-specific slope, and allowed causal keys satisfy $j\le i$.
Forbidden scores still require a separate mask.

<!-- visual:positional-relative-mechanisms -->
<figure class="learning-figure plot-panel" aria-labelledby="positional-relative-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="positional-relative-title">How can RoPE and ALiBi both encode relative position?</p>
	<svg viewBox="0 0 360 460" role="img" aria-labelledby="positional-relative-svg-title positional-relative-svg-desc">
		<title id="positional-relative-svg-title">RoPE changes query and key angles while ALiBi subtracts distance from attention scores</title>
		<desc id="positional-relative-svg-desc">The upper panel uses identical unit content vectors and one rotary frequency. At positions one and three, query and key are rotated to angles theta and three theta, leaving a two-theta gap. Shifting both to positions four and six rotates both vectors together and preserves the same two-theta gap and dot product cosine of two theta. The lower panel keeps content vectors unchanged. For a causal query at position six, keys at positions two, four, and six have distances four, two, and zero, so ALiBi adds score biases negative four s sub h, negative two s sub h, and zero before softmax.</desc>
		<defs><marker id="position-vector-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path></marker></defs>
		<rect class="viz-plot-bg" x="5" y="5" width="350" height="244" rx="4"></rect>
		<text class="viz-axis-label" x="16" y="25">RoPE · rotate Q and K before the dot product</text>
		<text class="viz-label" x="90" y="44" text-anchor="middle">positions (m, n) = (1, 3)</text>
		<circle class="viz-gridline" cx="90" cy="117" r="55"></circle><path class="viz-axis" d="M30 117H150 M90 177V57"></path>
		<path class="viz-baseline" d="M90 117L145 117"></path>
		<path d="M90 117L138 89" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;marker-end:url(#position-vector-arrow)"></path>
		<path d="M90 117L90 62" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;marker-end:url(#position-vector-arrow)"></path>
		<path class="viz-operating-guide" d="M124 97A39 39 0 0 0 90 78"></path>
		<text class="viz-callout" x="135" y="84">R₁q</text><text class="viz-callout" x="95" y="70">R₃k</text><text class="viz-label" x="111" y="91">2θ</text>
		<text class="viz-label" x="270" y="44" text-anchor="middle">shift both: (m, n) = (4, 6)</text>
		<circle class="viz-gridline" cx="270" cy="117" r="55"></circle><path class="viz-axis" d="M210 117H330 M270 177V57"></path>
		<path class="viz-baseline" d="M270 117L325 117"></path>
		<path d="M270 117L242 69" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;marker-end:url(#position-vector-arrow)"></path>
		<path d="M270 117L215 117" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;marker-end:url(#position-vector-arrow)"></path>
		<path class="viz-operating-guide" d="M250 83A39 39 0 0 0 231 117"></path>
		<text class="viz-callout" x="235" y="66">R₄q</text><text class="viz-callout" x="212" y="108">R₆k</text><text class="viz-label" x="235" y="96">2θ</text>
		<text class="viz-callout" x="180" y="197" text-anchor="middle">same offset → same angle gap → same positional effect</text>
		<text class="viz-label" x="180" y="218" text-anchor="middle">(Rₘq)ᵀ(Rₙk) = qᵀRₙ₋ₘk</text>
		<text class="viz-label" x="180" y="237" text-anchor="middle">for q = k = unit vector: score = cos((n − m)θ) = cos(2θ)</text>
		<rect class="viz-plot-bg" x="5" y="260" width="350" height="195" rx="4"></rect>
		<text class="viz-axis-label" x="16" y="281">ALiBi · keep Q and K unchanged; bias their score</text>
		<text class="viz-callout" x="18" y="309">causal query i = 6</text><rect class="viz-node--input" x="278" y="292" width="60" height="27" rx="3"></rect><text class="viz-callout" x="308" y="310" text-anchor="middle">query 6</text>
		<text class="viz-label" x="18" y="343">key j = 2</text><path d="M91 339H278" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#position-vector-arrow)"></path><text class="viz-callout" x="184" y="333" text-anchor="middle">distance 4 → bias −4sₕ</text>
		<text class="viz-label" x="18" y="376">key j = 4</text><path d="M184 372H278" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#position-vector-arrow)"></path><text class="viz-callout" x="231" y="366" text-anchor="middle">distance 2 → −2sₕ</text>
		<text class="viz-label" x="18" y="409">key j = 6</text><text class="viz-callout" x="266" y="409" text-anchor="end">distance 0 → 0</text><circle class="viz-operating-point" cx="278" cy="405" r="5"></circle>
		<text class="viz-axis-label" x="180" y="436" text-anchor="middle">content score − sₕ(i − j) → softmax</text>
		<text class="viz-label" x="180" y="449" text-anchor="middle">each head h has its own fixed positive slope sₕ</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> shifting both upper positions by three preserves their angle gap for fixed content vectors. ALiBi instead subtracts distance penalties from content scores before softmax. At slope 0.25, the lower symbolic biases become −1, −0.5, and 0. These are different operations, despite their common relative-offset dependence. Sources: <a href="https://arxiv.org/abs/2104.09864">RoFormer</a> and <a href="https://arxiv.org/abs/2108.12409">ALiBi</a>.</figcaption>
</figure>

### Compute and change the row

| Slope | Distances | Score biases | Softmax probabilities |
| --- | --- | --- | --- |
| $0.25$ | $[4,2,0]$ | $[-1,-0.5,0]$ | $[0.186324,0.307196,0.506480]$ |
| $0.5$ | $[4,2,0]$ | $[-2,-1,0]$ | $[0.090031,0.244728,0.665241]$ |

For the first row, exponentials are $[0.367879,0.606531,1]$, summing to $1.974410$.
The newest key is $e^1$ times as likely as the oldest because their content scores match.
Shifting query to $9$ and keys to $[5,7,9]$ preserves distances and probabilities for unchanged content.

**New task:** strengthen recency preference.
Double the slope and recompute softmax, as the second row shows.
**Use when:** deliberately adapting head distance preferences; changing deployed slopes changes the model.

**New task:** allow an encoder query at $6$ to read key $8$.
Use a bidirectional mask and the specified absolute-distance bias.
**Check:** slope $0.25$ gives $-0.25|6-8|=-0.5$, not $+0.5$ from misusing signed causal distance.
**Use when:** available source positions lie on both sides.

### Position-free / implicit position

A causal mask or recurrence can impose order without adding a positional vector.
Mixture-of-experts routing alone does not supply sequence position.

## What an interviewer expects you to say

Separate position representation, visibility, and context capacity.
**Failure:** a fixed slope is not a causal mask; unmasked future keys can receive an unintended positive signed-distance bias.

**Reconstruct:** can RoPE replace ALiBi unchanged because both use relative positions?

<details><summary>Answer</summary>

No. Rotation changes content dot products; ALiBi adds score biases.
The same trained weights need not preserve scores, probabilities, or useful behavior after substitution.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Common confusions

Relative scores do not guarantee long-context generalization.
Context capacity also depends on memory, compute, implementation limits, and training.
An unbounded position formula is not an empirical quality result.

## Long-context extension techniques

- [Position interpolation](https://arxiv.org/abs/2306.15595) rescales positions toward the trained range, with its specified continued training.
- NTK-aware variants change the frequency schedule; inspect the actual formula rather than assuming one universal implementation.
- [YaRN](https://arxiv.org/abs/2309.00071) changes interpolation and attention scaling under its chosen recipe.
- Continued long-context training changes the learning exposure and compute budget, not merely a position index limit.

Evaluate short-context retention, retrieval, reasoning, location sensitivity, latency, and memory separately.
One successful retrieval needle does not qualify every long-document task.

## Why interviewers ask

The useful test is whether you can locate the operation, reproduce the score change, and specify evidence for adaptation.

---

*Related: [Transformer architecture](/concepts/transformer-architecture/), [FlashAttention](/concepts/flashattention/), [KV cache](/concepts/kv-cache/).*

</div>
