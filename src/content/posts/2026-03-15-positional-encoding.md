---
title: "RoPE, ALiBi, and modern positional encodings"
description: "Modern LLMs usually replace sinusoidal positional encoding with RoPE, ALiBi, or related methods. Compare how they represent relative position and extrapolate."
date: "2026-03-15"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Positional encoding gives a transformer information about token *order*, since attention itself is permutation-invariant. Modern LLMs use **rotary position embeddings (RoPE)** or **ALiBi** instead of the original sinusoidal scheme, primarily for better long-context behavior.

The choice of positional encoding determines long-context performance, extrapolation beyond training length, and relative position representation. It's a small choice with outsized impact on production inference.

## The lineup

### Sinusoidal (original Transformer, 2017)

Add a fixed sinusoidal pattern to the input embeddings:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- Encodes absolute position.
- Allows some implicit relative-position learning through linear combinations of sinusoids.
- Limited extrapolation: at positions much beyond training length, behavior degrades.

### Learned absolute positional embeddings

A learnable embedding per position. Used in BERT, GPT-2.

- Each position gets its own embedding learned during training.
- Cannot extrapolate at all to lengths beyond training.
- Larger memory cost (one embedding per position).

### Relative positional encoding

Replace absolute positions with relative offsets. T5 introduced a simple bucketed bias added to attention scores.

- Better generalization than absolute.
- Several variants (T5, Transformer-XL, Shaw et al.).

### RoPE (Rotary Position Embeddings), the modern default

Don't add positional info to embeddings. Instead, *rotate* Q and K by position-dependent angles before the attention dot product.

For position `m`, the rotation matrix `R_m` is block-diagonal with 2x2 rotations of angles `m * theta_i` for each pair of dimensions. Then:
```
Q' = R_m * Q   (Q rotated by query position)
K' = R_n * K   (K rotated by key position)
```

The dot product `Q'^T * K' = Q^T * R_{n-m} * K` depends only on the *relative* position `n - m`, not on absolute positions. So RoPE is implicitly relative.

- Encodes relative position naturally.
- Plays well with FlashAttention (rotation is an element-wise op).
- Has no learned position-table cutoff, but unscaled use far beyond the training length can still degrade.
- Used in: LLaMA family, Mistral, Qwen, Gemma, most modern open LLMs.

Variants exist for context extension: **NTK-aware RoPE**: **YaRN**: **PI (Position Interpolation)**: rescale the rotation frequencies to handle longer-than-trained contexts.

### ALiBi (Attention with Linear Biases)

Add a position-dependent *bias* directly to attention scores:
```
score_h(i, j) = Q_i K_j^T / sqrt(d) - s_h * (i - j)
```
where `s_h > 0` is a fixed per-head slope and `i - j >= 0` is the distance from a causal query to an unmasked key. For bidirectional attention, use absolute distance.

- No additional parameters; no embedding modification.
- Implicitly relative.
- Excellent extrapolation to longer sequences than seen at training time.
- Used in: MPT, BLOOM, some BERT variants.

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
	<figcaption><strong>Read it this way:</strong> in the top panel, shift both positions by three: both vectors rotate together, so their angle gap remains <code>2θ</code> and the positional part of their dot product is unchanged. In the bottom panel, ALiBi never rotates the vectors; after the content dot product, it subtracts a larger score penalty from more distant keys. Both mechanisms depend on relative offset, but at different stages. This original schematic is checked against the <a href="https://arxiv.org/abs/2104.09864">RoFormer paper</a> and the <a href="https://arxiv.org/abs/2108.12409">ALiBi paper</a>.</figcaption>
</figure>

### Position-free / implicit position

Some recent architectures (some MoE variants, certain SSM-based models) avoid explicit positional encoding by relying on the recurrence or state dynamics to inject position. Less common in transformers proper.

## What an interviewer expects you to say

If asked about positional encoding:

1. Explain *why* attention needs positional info (permutation-invariance).
2. Mention the original sinusoidal scheme as a starting point.
3. State that modern LLMs use **RoPE or ALiBi**: not sinusoidal.
4. Explain RoPE's mechanism (rotation in 2D blocks; gives relative position implicitly).
5. Discuss the long-context extrapolation issue and the techniques (NTK-aware, YaRN, PI) for extending RoPE-trained models to longer contexts.

If you describe positional encoding in 2026 using only sinusoidal, you signal your knowledge stops in 2020.

## Common confusions

- **"Sinusoidal is the standard."** It was the standard in 2017-2019 and is now obsolete in production. RoPE is the standard since ~2021.
- **"Absolute vs relative positional encoding."** A meaningful distinction. RoPE and ALiBi are both relative.
- **"Positional encoding extends the context window."** No, the model architecture and training data extend the context window. PE choices affect *how gracefully* the model handles long contexts and whether it can extrapolate.
- **"YaRN is a different positional encoding."** YaRN is a specific *adaptation* of RoPE that extends context, not a separate scheme.

## Long-context extension techniques

A practical concern: most LLMs are pretrained at modest context (e.g., 8K) but production wants much longer (32K, 128K, 1M). Three families of fix:

1. **Position interpolation (PI)**: scale position indices down so that positions in the longer context map into the trained range, then fine-tune at the longer length.
2. **NTK-aware scaling**: scale RoPE frequencies non-uniformly so high-frequency dimensions are preserved (which matter for short-distance precision) and low-frequency dimensions are stretched. Better than naive PI.
3. **YaRN**: interpolate frequencies selectively by wavelength and adjust attention scale. It can improve zero-shot extension, while short continued training produces stronger results.
4. **Continued pretraining at long context**: the most reliable but expensive option. Used by Anthropic, OpenAI, etc.

Long-context extension is now standard interview territory because it's a commercial differentiator in 2026.

## Why interviewers ask

Positional encoding tests:
1. Whether you've kept up with transformer evolution since 2020.
2. Whether you understand attention's permutation-invariance and why position info is needed.
3. Whether you've handled long-context concerns in production.

A senior follow-up: "How would you extend a RoPE-trained model from 8K to 128K context?" Answer: NTK-aware scaling or YaRN, possibly followed by continued pretraining on long-context data; evaluate retrieval quality (needle-in-haystack) at the new length to verify. This is a standard 2026 problem and the answer signals production fluency.

---

*Related: [Transformer architecture](/concepts/transformer-architecture/), [FlashAttention](/concepts/flashattention/), [KV cache](/concepts/kv-cache/).*
