---
title: "Linear attention (Linformer, Performer, kernel methods)"
description: "Approximate the softmax attention matrix with a low-rank or kernel factorization so cost is linear in sequence length."
date: "2025-12-06"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Linear attention avoids materializing the $n \times n$ softmax matrix by projecting the sequence axis or approximating the softmax kernel with a low-dimensional feature map. The per-layer cost drops from $O(n^2 d)$ to $O(n k d)$ or $O(n r d)$ for $k, r \ll n$.

Sparse attention (see [sparse attention](/concepts/sparse-attention/)) keeps the softmax exact but on fewer pairs. Linformer instead uses a learned low-rank projection along the sequence dimension; Performer approximates the softmax kernel with random features and changes the order of multiplication.

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

Write $Q' = \phi(Q)$ and $K' = \phi(K)$. Because softmax attention is row-normalized, the approximation needs both a value summary and a normalizer. For query row $i$,

$$
\text{Attn}_i \approx
\frac{\phi(q_i)^\top (K'^\top V)}
     {\phi(q_i)^\top (K'^\top \mathbf{1})}.
$$

Compute the right-hand contractions first: $K'^\top V$ is $r \times d$ and $K'^\top \mathbf{1}$ is $r \times 1$. Each transformed query reads these summaries, so no $n \times n$ matrix is formed. Cost: $O(n r d)$, linear in $n$, and the feature map works for arbitrary $n$ at inference (no fixed sequence-length projection).

<!-- visual:linear-attention-two-escape-routes -->
<figure class="learning-figure" aria-labelledby="linear-attention-routes-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="linear-attention-routes-title">Where does each method remove the n × n matrix?</p>
	<div class="visual-grid--two" role="group" aria-label="Two ways linear attention avoids all query-key pairs">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 320 275" role="img" aria-labelledby="linformer-route-title linformer-route-desc">
				<title id="linformer-route-title">Linformer compresses the sequence axis before computing attention scores</title>
				<desc id="linformer-route-desc">Keys and values each start with shape n by d. Learned projections E and F reduce their token dimension from n to k, producing E K and F V with shape k by d. Queries retain shape n by d. Queries meet only the k projected keys, producing n by k weights, which mix the k projected values into an n by d output. An n by n score matrix is never formed.</desc>
				<defs><style>text.viz-label,text.viz-axis-label,text.viz-callout,text.viz-node-value{font-size:12px}</style><marker id="linformer-route-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" style="fill:var(--viz-edge)"></path></marker></defs>
				<text class="viz-axis-label" x="10" y="17">LINFORMER · COMPRESS TOKENS n → k</text>
				<rect class="viz-node viz-node--input" x="10" y="34" width="78" height="38" rx="4"></rect>
				<text class="viz-node-label" x="49" y="51">K</text><text class="viz-node-value" x="49" y="65">n × d</text>
				<rect class="viz-node viz-node--input" x="10" y="88" width="78" height="38" rx="4"></rect>
				<text class="viz-node-label" x="49" y="105">V</text><text class="viz-node-value" x="49" y="119">n × d</text>
				<path d="M88 53H117M88 107H117" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#linformer-route-arrow)"></path>
				<text class="viz-label" x="103" y="45" text-anchor="middle">E</text><text class="viz-label" x="103" y="99" text-anchor="middle">F</text>
				<rect class="viz-node viz-node--focus" x="120" y="34" width="82" height="38" rx="4"></rect>
				<text class="viz-callout" x="161" y="51" text-anchor="middle">EK</text><text class="viz-node-value" x="161" y="65">k × d</text>
				<rect class="viz-node viz-node--focus" x="120" y="88" width="82" height="38" rx="4"></rect>
				<text class="viz-callout" x="161" y="105" text-anchor="middle">FV</text><text class="viz-node-value" x="161" y="119">k × d</text>
				<rect class="viz-node viz-node--input" x="10" y="153" width="78" height="38" rx="4"></rect>
				<text class="viz-node-label" x="49" y="170">Q</text><text class="viz-node-value" x="49" y="184">n × d</text>
				<path d="M88 172H122M161 72V145" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#linformer-route-arrow)"></path>
				<rect class="viz-node viz-node--focus" x="125" y="145" width="94" height="54" rx="4"></rect>
				<text class="viz-callout" x="172" y="165" text-anchor="middle">softmax</text><text class="viz-node-value" x="172" y="181">Q(EK)ᵀ: n × k</text>
				<path d="M202 107H239V172H222M219 172H246" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#linformer-route-arrow)"></path>
				<rect class="viz-node viz-node--output" x="249" y="145" width="61" height="54" rx="4"></rect>
				<text class="viz-callout" x="279.5" y="165" text-anchor="middle">output</text><text class="viz-node-value" x="279.5" y="181">n × d</text>
				<path class="viz-operating-guide" d="M24 229H128"></path><path d="M60 216L91 244M91 216L60 244" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-axis-label" x="143" y="233">n × n scores</text><text class="viz-label" x="143" y="250">never created</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 320 275" role="img" aria-labelledby="performer-route-title performer-route-desc">
				<title id="performer-route-title">Performer summarizes transformed keys before queries read them</title>
				<desc id="performer-route-desc">Transformed keys K prime have shape n by r and values have shape n by d. Contracting across n first creates a value summary K prime transpose V of shape r by d and a normalizer K prime transpose one of shape r by one. A transformed query row with shape one by r reads both summaries and divides the numerator by the normalizer to produce one output row with shape one by d. Repeating this per query never forms n by n scores.</desc>
				<defs><style>text.viz-label,text.viz-axis-label,text.viz-callout,text.viz-node-value{font-size:12px}</style><marker id="performer-route-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" style="fill:var(--viz-edge)"></path></marker></defs>
				<text class="viz-axis-label" x="10" y="17">PERFORMER · SUMMARIZE FEATURES r</text>
				<rect class="viz-node viz-node--input" x="10" y="34" width="84" height="38" rx="4"></rect>
				<text class="viz-callout" x="52" y="51" text-anchor="middle">K′ = φ(K)</text><text class="viz-node-value" x="52" y="65">n × r</text>
				<rect class="viz-node viz-node--input" x="10" y="88" width="84" height="38" rx="4"></rect>
				<text class="viz-node-label" x="52" y="105">V</text><text class="viz-node-value" x="52" y="119">n × d</text>
				<path d="M94 53H118M94 53L118 107M94 107L118 53" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#performer-route-arrow)"></path>
				<rect class="viz-node viz-node--focus" x="121" y="34" width="112" height="38" rx="4"></rect>
				<text class="viz-callout" x="177" y="51" text-anchor="middle">K′ᵀV first</text><text class="viz-node-value" x="177" y="65">r × d summary</text>
				<rect class="viz-node viz-node--focus" x="121" y="88" width="112" height="38" rx="4"></rect>
				<text class="viz-callout" x="177" y="105" text-anchor="middle">K′ᵀ1 first</text><text class="viz-node-value" x="177" y="119">r × 1 normalizer</text>
				<rect class="viz-node viz-node--input" x="10" y="153" width="102" height="38" rx="4"></rect>
				<text class="viz-callout" x="61" y="170" text-anchor="middle">φ(qᵢ)ᵀ</text><text class="viz-node-value" x="61" y="184">1 × r</text>
				<path d="M112 172H140M177 72V143M177 126V143" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#performer-route-arrow)"></path>
				<rect class="viz-node viz-node--focus" x="143" y="143" width="90" height="58" rx="4"></rect>
				<text class="viz-callout" x="188" y="164" text-anchor="middle">read both</text><text class="viz-node-value" x="188" y="180">numerator</text><text class="viz-node-value" x="188" y="194">÷ normalizer</text>
				<path d="M233 172H246" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#performer-route-arrow)"></path>
				<rect class="viz-node viz-node--output" x="249" y="145" width="61" height="54" rx="4"></rect>
				<text class="viz-callout" x="279.5" y="165" text-anchor="middle">oᵢ</text><text class="viz-node-value" x="279.5" y="181">1 × d</text>
				<path class="viz-operating-guide" d="M24 229H128"></path><path d="M60 216L91 244M91 216L60 244" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-axis-label" x="143" y="233">n × n scores</text><text class="viz-label" x="143" y="250">never created</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> follow the shapes, not the colors. Linformer reduces the token axis from <var>n</var> to <var>k</var> before queries meet keys, so its weight matrix is <var>n</var> × <var>k</var>. Performer keeps all <var>n</var> tokens but contracts transformed keys with values first, leaving an <var>r</var> × <var>d</var> summary and an <var>r</var> × 1 normalizer for each query to read. Both routes avoid <var>n</var> × <var>n</var>, but by compressing different axes. Original schematic based on <a href="https://arxiv.org/abs/2006.04768">Linformer</a> and <a href="https://arxiv.org/abs/2009.14794">Performer</a>.</figcaption>
</figure>

## When to use linear attention in 2026

- Sequence length $n \gg 32{,}000$ where FlashAttention is still too slow or doesn't fit memory.
- Encoder-only models on very long inputs.
- Real-time inference with strict latency budgets and tolerable quality loss.

For chat-style decoder LLMs, dense attention with FlashAttention + GQA + KV cache remains the production default.

## Common pitfalls

- **Comparing FLOPs without measuring wall-clock.** Linear attention's $O(n)$ scaling only beats $O(n^2)$ FlashAttention at large $n$; the crossover is implementation-dependent and often higher than naive analysis suggests.
- **Forgetting the constants.** Linformer's $k$ and Performer's $r$ may need to be hundreds for good quality, so the linear scaling has a large constant.
- **Assuming all softmax-replacement schemes preserve the autoregressive mask trivially.** Kernelized attention requires careful handling for causal masking (recursive cumulative sums).
