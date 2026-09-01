---
title: "Sparse attention (BigBird, Longformer)"
description: "Replace the dense n×n attention mask with a sparse pattern that has O(n) non-zeros while preserving information flow across the full sequence."
date: "2026-02-08"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Sparse attention is exact attention restricted to a structured sparse mask, so cost scales as $O(n \cdot k)$ for some constant $k \ll n$ instead of $O(n^2)$, while the mask is designed so information can still propagate to all positions in a small number of layers.

Dense self-attention costs $O(n^2 \cdot d)$ in compute and $O(n^2)$ in memory. For long inputs (long documents, long-form audio, code repos) this becomes the binding constraint. Empirically most attention weights are near zero. So the dense matrix is wasteful.

Sparse attention was the dominant pre-2023 approach to long context. It is still relevant for encoder-only long-document models (clinical notes, legal contracts, scientific papers).

## The two canonical patterns

<!-- visual:sparse-attention-reach-patterns -->
<figure class="learning-figure" aria-labelledby="sparse-attention-reach-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="sparse-attention-reach-title">Trace how a sparse mask keeps distant tokens reachable without giving every token every edge.</p>
	<div class="visual-grid--two" role="group" aria-label="Longformer and BigBird sparse attention routes">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="longformer-route-title longformer-route-desc">
				<title id="longformer-route-title">Longformer combines a local window with a global hub</title>
				<desc id="longformer-route-desc">Seven token positions are shown. Query token four has solid local links to its immediate neighbors. A designated global token G connects to every position. Information from distant token seven can enter G in layer one and move from G to query token four in layer two.</desc>
				<defs><marker id="longformer-route-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" style="fill:var(--viz-edge)"></path></marker></defs>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="197" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">LONGFORMER · WINDOW + GLOBAL</text>
				<path d="M132 88L96 88M168 88L204 88" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path>
				<path d="M42 74L78 74M42 74L114 74M42 74L150 74M42 74L186 74M42 74L222 74M42 74L258 74" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:3 3"></path>
				<g>
					<rect class="viz-node viz-node--output" x="24" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="60" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="96" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node viz-node--input" x="132" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="168" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="204" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="240" y="70" width="36" height="36" rx="4"></rect>
					<text class="viz-node-label" x="42" y="93">G</text><text class="viz-node-label" x="78" y="93">2</text><text class="viz-node-label" x="114" y="93">3</text><text class="viz-node-label" x="150" y="93">q</text><text class="viz-node-label" x="186" y="93">5</text><text class="viz-node-label" x="222" y="93">6</text><text class="viz-node-label" x="258" y="93">7</text>
				</g>
				<text class="viz-label" x="150" y="126" text-anchor="middle">solid = q's local window</text>
				<text class="viz-label" x="150" y="143" text-anchor="middle">dashed = global G attends every token</text>
				<path d="M258 165H51" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8;marker-end:url(#longformer-route-arrow)"></path>
				<path d="M42 173Q42 202 75 202H150V178" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8;marker-end:url(#longformer-route-arrow)"></path>
				<text class="viz-callout" x="207" y="160" text-anchor="middle">layer 1: 7 → G</text>
				<text class="viz-callout" x="105" y="218" text-anchor="middle">layer 2: G → q</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="bigbird-route-title bigbird-route-desc">
				<title id="bigbird-route-title">BigBird adds random shortcuts to local and global links</title>
				<desc id="bigbird-route-desc">The same seven positions retain the local window and global token. Query token four also has a separately labelled random link to token seven. With fixed local width, random links, and global tokens per query, the number of allowed pairs grows linearly with sequence length rather than quadratically.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="197" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">BIGBIRD · WINDOW + GLOBAL + RANDOM</text>
				<path d="M132 88L96 88M168 88L204 88" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path>
				<path d="M42 74L78 74M42 74L114 74M42 74L150 74M42 74L186 74M42 74L222 74M42 74L258 74" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:3 3"></path>
				<path d="M150 70Q204 34 258 70" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-dasharray:7 3"></path>
				<g>
					<rect class="viz-node viz-node--output" x="24" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="60" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="96" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node viz-node--input" x="132" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="168" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="204" y="70" width="36" height="36" rx="4"></rect>
					<rect class="viz-node" x="240" y="70" width="36" height="36" rx="4"></rect>
					<text class="viz-node-label" x="42" y="93">G</text><text class="viz-node-label" x="78" y="93">2</text><text class="viz-node-label" x="114" y="93">3</text><text class="viz-node-label" x="150" y="93">q</text><text class="viz-node-label" x="186" y="93">5</text><text class="viz-node-label" x="222" y="93">6</text><text class="viz-node-label" x="258" y="93">7</text>
				</g>
				<text class="viz-callout" x="204" y="42" text-anchor="middle">random shortcut</text>
				<text class="viz-label" x="150" y="126" text-anchor="middle">solid local · dotted global · long-dash random</text>
				<rect class="viz-node" x="31" y="150" width="238" height="48" rx="4"></rect>
				<text class="viz-axis-label" x="150" y="169" text-anchor="middle">FIXED EDGES PER QUERY</text>
				<text class="viz-callout" x="150" y="188" text-anchor="middle">w local + r random + g global</text>
				<text class="viz-axis-label" x="150" y="216" text-anchor="middle">n QUERIES × CONSTANT EDGES = O(n)</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> start at <code>q</code>. Solid links preserve nearby context; the dotted global hub gives distant information a two-layer route; BigBird's long-dashed random link adds another shortcut. Each query keeps only a fixed number of these links, so the mask grows as <em>n</em>, not <em>n²</em>.</figcaption>
</figure>

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
- **Confusing sparse with low-rank.** Sparse keeps the softmax exact but on fewer pairs; low-rank methods (Linformer, Performer. See [linear attention](/concepts/linear-attention/)) approximate the softmax matrix itself.
- **Picking $w$ too small.** With window 32 and 24 layers, information at position 0 cannot reach position 5000 in one forward pass. Either widen the window or add global tokens.
