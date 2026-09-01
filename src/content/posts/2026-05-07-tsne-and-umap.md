---
title: "t-SNE and UMAP: nonlinear dimensionality reduction"
description: "Both project high-dimensional data to 2D for visualization by preserving local neighborhoods. Both are easy to misread. Know what they show and what they hide."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**t-SNE** ([van der Maaten & Hinton, 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html)) and **UMAP** ([McInnes et al., 2018](https://arxiv.org/abs/1802.03426)) embed high-dimensional points into 2 or 3 dimensions while preserving local neighborhood structure. The default tools for "what does this embedding space look like" plots.

Linear projections (PCA) preserve global variance but smear local structure. For high-dimensional embeddings (transformer activations, sentence embeddings, single-cell genomics), the interesting structure is local: which points cluster together, which categories are separable. t-SNE and UMAP optimize for that locally and produce maps that show the cluster structure clearly.

Almost every embedding visualization you have seen in a paper since 2015 is one of these two.

## What t-SNE optimizes

For each high-dimensional point $x_i$, define a probability distribution over neighbors using a Gaussian:

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2 \sigma_i^2)}{\sum_{k \ne i} \exp(-\|x_i - x_k\|^2 / 2 \sigma_i^2)},
$$

with $\sigma_i$ tuned per point so that the entropy of $p_{\cdot \mid i}$ matches a target **perplexity** (typically 30, an effective neighborhood size).

In 2D, define a heavy-tailed (Student-$t$) distribution:

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|y_k - y_l\|^2)^{-1}}.
$$

Minimize the KL divergence $\text{KL}(P \| Q)$ via gradient descent on $y_i$. The heavy tail in $q$ pushes far-apart points further apart, opening visible gaps between clusters.

## What UMAP optimizes

UMAP builds a fuzzy simplicial set (a weighted graph) of the high-dimensional data using each point's $k$ nearest neighbors. It does the same in low dimension and minimizes a cross-entropy between the two graphs. Faster than t-SNE, scales to millions of points, often gives slightly better global structure.

The math is more involved (it involves Riemannian metrics and category theory in the original paper), but operationally UMAP is "t-SNE on a sparse k-NN graph with a different loss."

## What both preserve and what they don't

**Preserve well**:
- Local neighborhood: which points are close to which.
- Cluster identity: separable groups remain separable.

**Do not preserve**:
- **Distances between clusters.** Cluster $A$ being twice as far from cluster $B$ as from cluster $C$ in the t-SNE plot tells you almost nothing about the high-dimensional reality.
- **Cluster sizes.** A small dense cluster and a large diffuse one can render the same size.
- **Densities.** UMAP and t-SNE both equalize density to some extent.

**Learning objective:** distinguish the local-neighborhood relationships these methods are designed to preserve from inter-cluster gaps and rendered cluster footprints that the 2D map does not reliably preserve.

<!-- visual:nonlinear-embedding-local-vs-global -->
<figure class="learning-figure" aria-labelledby="nonlinear-embedding-title">
	<p class="visual-kicker">Read neighborhoods, not geography</p>
	<p class="visual-title" id="nonlinear-embedding-title">What can stay true when the 2D map changes shape?</p>
	<div class="visual-grid--two">
		<section class="visual-panel plot-panel" aria-labelledby="embedding-layout-one-title">
			<h4 id="embedding-layout-one-title">Plausible layout 1</h4>
			<p>Selected local neighbor links are solid; global gaps are dashed.</p>
			<svg viewBox="0 0 300 240" role="img" aria-labelledby="embedding-layout-one-svg-title embedding-layout-one-svg-desc">
				<title id="embedding-layout-one-svg-title">Three groups in the first schematic embedding layout</title>
				<desc id="embedding-layout-one-svg-desc">Group A uses four circles connected by solid local neighbor links, group B uses four squares with the same pattern of local links, and group C uses four diamonds with the same pattern. A and B appear close and compact, while C appears far away. Dashed guides mark these global gaps as untrusted.</desc>
				<rect class="viz-plot-bg" x="5" y="5" width="290" height="210" rx="3"></rect>
				<path d="M48 72L70 55L87 80L64 94L48 72M70 55L64 94M48 72L87 80" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<circle class="viz-operating-point" cx="48" cy="72" r="5"></circle>
				<circle class="viz-operating-point" cx="70" cy="55" r="5"></circle>
				<circle class="viz-operating-point" cx="87" cy="80" r="5"></circle>
				<circle class="viz-operating-point" cx="64" cy="94" r="5"></circle>
				<text class="viz-callout" x="65" y="119" text-anchor="middle">A · circles</text>
				<path d="M128 72L151 57L168 82L144 96L128 72M151 57L144 96M128 72L168 82" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<rect class="viz-node viz-node--input" x="123" y="67" width="10" height="10" rx="1"></rect>
				<rect class="viz-node viz-node--input" x="146" y="52" width="10" height="10" rx="1"></rect>
				<rect class="viz-node viz-node--input" x="163" y="77" width="10" height="10" rx="1"></rect>
				<rect class="viz-node viz-node--input" x="139" y="91" width="10" height="10" rx="1"></rect>
				<text class="viz-callout" x="147" y="119" text-anchor="middle">B · squares</text>
				<path d="M217 156L239 138L259 163L234 181L217 156M239 138L234 181M217 156L259 163" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<path class="viz-node viz-node--output" d="M217 150L223 156L217 162L211 156Z"></path>
				<path class="viz-node viz-node--output" d="M239 132L245 138L239 144L233 138Z"></path>
				<path class="viz-node viz-node--output" d="M259 157L265 163L259 169L253 163Z"></path>
				<path class="viz-node viz-node--output" d="M234 175L240 181L234 187L228 181Z"></path>
				<text class="viz-callout" x="238" y="205" text-anchor="middle">C · diamonds</text>
				<path class="viz-operating-guide" d="M92 41H122"></path>
				<text class="viz-label" x="107" y="32" text-anchor="middle">looks close</text>
				<path class="viz-operating-guide" d="M178 119L213 145"></path>
				<text class="viz-label" x="205" y="123" text-anchor="middle">looks far</text>
				<text class="viz-axis-label" x="10" y="232">SCHEMATIC: NOT MEASURED OUTPUT</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel" aria-labelledby="embedding-layout-two-title">
			<h4 id="embedding-layout-two-title">Plausible layout 2</h4>
			<p>The same selected neighbor memberships survive a different map.</p>
			<svg viewBox="0 0 300 240" role="img" aria-labelledby="embedding-layout-two-svg-title embedding-layout-two-svg-desc">
				<title id="embedding-layout-two-svg-title">The same three groups in a different schematic embedding layout</title>
				<desc id="embedding-layout-two-svg-desc">The circles, squares, and diamonds retain the same selected solid local neighbor connections as in the first panel. Group A now has a wider footprint, group B appears far from A, and group C appears close to A. These changed global gaps and footprints are marked as untrusted.</desc>
				<rect class="viz-plot-bg" x="5" y="5" width="290" height="210" rx="3"></rect>
				<path d="M34 74L70 38L107 76L68 112L34 74M70 38L68 112M34 74L107 76" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<circle class="viz-operating-point" cx="34" cy="74" r="5"></circle>
				<circle class="viz-operating-point" cx="70" cy="38" r="5"></circle>
				<circle class="viz-operating-point" cx="107" cy="76" r="5"></circle>
				<circle class="viz-operating-point" cx="68" cy="112" r="5"></circle>
				<text class="viz-callout" x="70" y="132" text-anchor="middle">A · circles</text>
				<path d="M228 37L243 27L255 43L240 52L228 37M243 27L240 52M228 37L255 43" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<rect class="viz-node viz-node--input" x="223" y="32" width="10" height="10" rx="1"></rect>
				<rect class="viz-node viz-node--input" x="238" y="22" width="10" height="10" rx="1"></rect>
				<rect class="viz-node viz-node--input" x="250" y="38" width="10" height="10" rx="1"></rect>
				<rect class="viz-node viz-node--input" x="235" y="47" width="10" height="10" rx="1"></rect>
				<text class="viz-callout" x="242" y="75" text-anchor="middle">B · squares</text>
				<path d="M126 159L148 141L168 166L143 184L126 159M148 141L143 184M126 159L168 166" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<path class="viz-node viz-node--output" d="M126 153L132 159L126 165L120 159Z"></path>
				<path class="viz-node viz-node--output" d="M148 135L154 141L148 147L142 141Z"></path>
				<path class="viz-node viz-node--output" d="M168 160L174 166L168 172L162 166Z"></path>
				<path class="viz-node viz-node--output" d="M143 178L149 184L143 190L137 184Z"></path>
				<text class="viz-callout" x="146" y="207" text-anchor="middle">C · diamonds</text>
				<path class="viz-operating-guide" d="M112 33H216"></path>
				<text class="viz-label" x="164" y="24" text-anchor="middle">now looks far</text>
				<path class="viz-operating-guide" d="M94 124L120 151"></path>
				<text class="viz-label" x="129" y="128" text-anchor="middle">now looks close</text>
				<text class="viz-axis-label" x="10" y="232">SCHEMATIC: SAME LOCAL LINKS</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare shape-coded groups across panels. The solid within-group links encode the same selected local-neighbor relationships, which are the evidence these methods emphasize. The dashed inter-group gaps reverse, and A's footprint expands while B's contracts, so do not interpret gap length, cluster area, or density as measurements of the original space. This original schematic is not algorithm output; its interpretation is checked against the <a href="https://www.jmlr.org/papers/v9/vandermaaten08a.html">t-SNE paper</a>, the <a href="https://arxiv.org/abs/1802.03426">UMAP paper</a>, and <a href="https://distill.pub/2016/misread-tsne/">Distill's t-SNE interpretation guide</a>.</figcaption>
</figure>

## The hyperparameters that change everything

**t-SNE**:
- **Perplexity** (5 to 50 typical). Effective neighborhood size. Small perplexity captures fine structure; large perplexity captures broader patterns. Always plot multiple perplexities ([Wattenberg et al., 2016](https://distill.pub/2016/misread-tsne/)).
- **Iterations** (1000+). Under-converged plots can show fake structure.
- **Initialization** (random vs PCA). PCA init gives more reproducible global layout.

**UMAP**:
- **n_neighbors** (15 to 50 typical). Local vs global tradeoff.
- **min_dist** (0.0 to 0.5). How tightly points are packed.
- **metric**. Cosine for embeddings, Euclidean for raw features.

## When to use which

| Use case | Tool |
|---|---|
| Datasets up to ~10k points, careful interpretation | t-SNE |
| Datasets above 100k points, speed matters | UMAP |
| Need approximate global structure | UMAP |
| Reproducible plots across runs | UMAP with fixed seed (t-SNE is also seed-dependent but more sensitive) |

## Common pitfalls

- **Reading distance between clusters as meaningful.** It is not.
- **Running with default hyperparameters and never sweeping.** Conclusions can flip with perplexity or n_neighbors.
- **Using t-SNE for downstream features.** It is for visualization only; the embedding is not a meaningful low-dim representation.
- **Forgetting that the seed matters.** Always report it. Cross-check with multiple seeds before drawing conclusions.
- **Using Euclidean distance on raw embeddings**. Most modern embeddings are designed for cosine; pass `metric="cosine"`.

## Related

- [SVD and PCA](/concepts/svd-and-pca/).
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity/).
