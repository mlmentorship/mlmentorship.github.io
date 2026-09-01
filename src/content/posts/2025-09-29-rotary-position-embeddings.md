---
title: "Rotary position embeddings (RoPE)"
description: "The dominant position encoding for modern LLMs. Encodes relative position by rotating Q and K in 2D subspaces and supports several context-extension methods."
date: "2025-09-29"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

RoPE encodes token position by rotating each pair of dimensions of the query and key vectors by an angle proportional to position, so that the inner product $Q_m^\top K_n$ depends only on the relative offset $m - n$.

Standard absolute position embeddings (sinusoidal in original transformer; learned in BERT/GPT-2) are added to token embeddings at the input. They couple position with content additively and don't extend cleanly past the training context.

RoPE [(Su et al., 2021)](https://arxiv.org/abs/2104.09864) is a multiplicative scheme applied inside attention. It became the default in modern decoder LLMs: Llama 1/2/3, Mistral, Qwen, DeepSeek, GPT-NeoX. Its rotation formula has no learned position-table cutoff, although using it far beyond the trained context can still degrade quality. Context-extension methods such as NTK-aware scaling, YaRN, and position interpolation modify the frequencies or the positions supplied to RoPE.

## The mechanism

Split the head dimension $d$ into $d/2$ pairs. For each pair $i \in \{0, \dots, d/2 - 1\}$ pick a frequency $\theta_i = 10000^{-2i/d}$ (same base as sinusoidal). For a token at position $m$, rotate the $i$-th 2D pair by angle $m\theta_i$:

$$
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

Apply the same rotation to keys (with their position $n$). The inner product satisfies $\langle Q'_m, K'_n \rangle = f(Q_m, K_n, m - n)$. Depends only on the relative offset.

In code, RoPE is implemented as elementwise multiplies with precomputed `cos` and `sin` tables; no extra parameters.

<!-- visual:rope-relative-rotation -->
<figure class="learning-figure plot-panel" aria-labelledby="rope-relative-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="rope-relative-title">Why do absolute rotations produce a relative attention score?</p>
	<svg viewBox="0 0 360 308" role="img" aria-labelledby="rope-relative-svg-title rope-relative-svg-desc">
		<title id="rope-relative-svg-title">A common position shift preserves the angle between rotary query and key vectors</title>
		<desc id="rope-relative-svg-desc">Two coordinate circles compare one query-key dimension pair at positions one and three with the same pair shifted to positions four and six. In the first circle, the query is rotated by theta and the key by three theta, leaving a two-theta angle gap. In the second, both positions increase by three, so the query and key rotate together and retain the same two-theta gap. The identity R sub m transpose R sub n equals R sub n minus m explains why the dot product depends on relative offset. Values are not rotated.</desc>
		<defs><marker id="rope-vector-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path></marker></defs>
		<rect class="viz-plot-bg" x="5" y="5" width="350" height="298" rx="4"></rect>
		<text class="viz-axis-label" x="90" y="28" text-anchor="middle">positions (m, n) = (1, 3)</text>
		<circle class="viz-gridline" cx="90" cy="112" r="57"></circle><path class="viz-axis" d="M28 112H152 M90 174V50"></path>
		<path d="M90 112L139 84" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;marker-end:url(#rope-vector-arrow)"></path>
		<path d="M90 112L90 55" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-dasharray:5 3;marker-end:url(#rope-vector-arrow)"></path>
		<path class="viz-operating-guide" d="M124 92A39 39 0 0 0 90 73"></path>
		<text class="viz-callout" x="139" y="78">R₁q</text><text class="viz-callout" x="96" y="61">R₃k</text><text class="viz-label" x="108" y="85">2θ</text>
		<text class="viz-axis-label" x="270" y="28" text-anchor="middle">shift both: (4, 6)</text>
		<circle class="viz-gridline" cx="270" cy="112" r="57"></circle><path class="viz-axis" d="M208 112H332 M270 174V50"></path>
		<path d="M270 112L242 63" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;marker-end:url(#rope-vector-arrow)"></path>
		<path d="M270 112L213 112" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-dasharray:5 3;marker-end:url(#rope-vector-arrow)"></path>
		<path class="viz-operating-guide" d="M250 78A39 39 0 0 0 231 112"></path>
		<text class="viz-callout" x="232" y="61">R₄q</text><text class="viz-callout" x="212" y="103">R₆k</text><text class="viz-label" x="235" y="88">2θ</text>
		<text class="viz-callout" x="180" y="199" text-anchor="middle">common shift rotates both vectors together</text>
		<text class="viz-axis-label" x="180" y="226" text-anchor="middle">(Rₘq)ᵀ(Rₙk) = qᵀRₘᵀRₙk = qᵀRₙ₋ₘk</text>
		<text class="viz-label" x="180" y="248" text-anchor="middle">same n − m = 2 → same angle gap → same positional effect</text>
		<path class="viz-baseline" d="M25 267H335"></path>
		<text class="viz-callout" x="180" y="287" text-anchor="middle">Q and K rotate before their dot product · V stays unchanged</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> compare the two circles. Adding three to both token positions turns <code>(1, 3)</code> into <code>(4, 6)</code> and rotates both vectors by the same extra angle. Their absolute directions change, but their <code>2θ</code> separation does not. Algebraically, <code>RₘᵀRₙ = Rₙ₋ₘ</code>, so this pair's contribution to the attention score depends on <code>n − m</code>. Solid Q and dashed K arrows keep the distinction visible without color. Original schematic, checked against the <a href="https://arxiv.org/abs/2104.09864">RoFormer paper</a>.</figcaption>
</figure>

## Why it works

- **Relative**: attention scores depend on $m - n$, not absolute positions, matching what attention should care about.
- **Distance-sensitive**: high-frequency pairs ($i$ small) rotate fast. Each pair's contribution oscillates with distance; across many frequencies, their combined relative-position signal tends to weaken as distance grows.
- **Length-flexible formula**: rotations are defined at any position, so RoPE has no learned-table cutoff, but reliable use beyond the training range still requires care.

## Context extension

To run a RoPE model past its training length:
- **Position interpolation** [(Chen et al., 2023)](https://arxiv.org/abs/2306.15595): linearly compress positions so the new max length maps to the original training range.
- **NTK-aware scaling**: increase the RoPE base $10000$ to a larger value so high-frequency components don't alias.
- **YaRN** [(Peng et al., 2023)](https://arxiv.org/abs/2309.00071): per-frequency interpolation tuned by training length statistics.

They alter the position-to-angle mapping without changing the attention architecture.

## Common pitfalls

- **Applying RoPE to V.** Only Q and K are rotated; V is not.
- **Confusing with ALiBi.** ALiBi adds a fixed slope to attention scores; RoPE rotates Q/K. Both encode relative position but are different mechanisms.
- **Forgetting the base when extending context.** Naively running a 4K model at 32K without scaling produces garbage past 4K.
