---
title: "SVD and PCA"
description: "The singular value decomposition factorizes any matrix into rotation × stretching × rotation. PCA is SVD applied to mean-centered data."
date: "2025-09-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Every real matrix $A \in \mathbb{R}^{m \times n}$ admits the factorization $A = U \Sigma V^\top$ where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal and $\Sigma$ is diagonal with non-negative entries (singular values). **PCA** is SVD applied to a mean-centered data matrix.

SVD is the universal matrix factorization. It exists for every matrix, even rectangular and rank-deficient ones. Reading off properties from the SVD answers "what does this matrix do?": singular values give scaling factors, $V$ gives input directions, $U$ gives output directions.

PCA is the canonical use of SVD: project data onto the directions of largest variance to get a low-dimensional representation that preserves as much information as possible.

## The decomposition

$A = U \Sigma V^\top$:

- $V$'s columns are orthonormal input directions; the first $r$ span the row space and the rest span the nullspace.
- $U$'s columns are orthonormal output directions; the first $r$ span the column space and the rest span the left nullspace.
- $\Sigma$'s diagonal entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ are the singular values (how much each input direction is stretched into its output direction).

Geometrically: any linear map is "rotate the input, stretch axis-by-axis, rotate the output." That's it.

The rank of $A$ is the number of non-zero singular values. For a full-rank map, the 2-norm condition number is $\sigma_1 / \sigma_{\min}$. A rank-deficient map has infinite condition number; $\sigma_1 / \sigma_r$ instead describes conditioning after restricting to its rank-$r$ subspace.

## Truncated SVD and low-rank approximation

The best rank-$k$ approximation of $A$ in Frobenius (or spectral) norm is

$$
A_k = U_k \Sigma_k V_k^\top
$$

where $U_k, V_k$ keep the first $k$ columns and $\Sigma_k$ keeps the first $k$ singular values (Eckart–Young theorem). Used in: dimensionality reduction, image compression, embedding regularization, low-rank LoRA fine-tuning.

## PCA as SVD

Given a data matrix $X \in \mathbb{R}^{n \times d}$ ($n$ samples, $d$ features):

1. Mean-center: $\tilde X = X - \bar x$.
2. Compute SVD: $\tilde X = U \Sigma V^\top$.
3. The columns of $V$ are the **principal components** (directions of maximum variance in feature space).
4. The variance along the $i$-th component is $\sigma_i^2 / (n - 1)$.
5. Project to $k$ dimensions: $Z = \tilde X V_k = U_k \Sigma_k$.

Equivalent formulation: PCA = eigendecomposition of the sample covariance $\tilde X^\top \tilde X / (n-1)$. SVD is numerically more stable.

<!-- visual:pca-rank-one-projection -->
<figure class="learning-figure" aria-labelledby="pca-projection-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="pca-projection-title">Rank-1 PCA keeps position along PC1 and discards the perpendicular residual.</p>
	<div class="visual-panel plot-panel visual-scroll visual-wide">
		<svg viewBox="0 0 640 360" role="img" aria-labelledby="pca-svg-title pca-svg-desc">
			<title id="pca-svg-title">Six centered samples projected onto their first principal component</title>
			<desc id="pca-svg-desc">Six circular sample points form a long, narrow cloud around the mean. PC1 follows the cloud's longest direction and PC2 is perpendicular. Dashed perpendicular segments connect every sample to a diamond on PC1. Each diamond is that sample's rank-1 reconstruction; each dashed segment is the discarded residual.</desc>
			<rect class="viz-plot-bg" x="20" y="10" width="600" height="330" rx="3"></rect>
			<circle class="viz-node" cx="52" cy="32" r="6"></circle>
			<text class="viz-label" x="65" y="36">centered sample x</text>
			<path class="viz-operating-point" d="M207 26 L213 32 L207 38 L201 32 Z"></path>
			<text class="viz-label" x="220" y="36">rank-1 reconstruction x̂</text>
			<path class="viz-baseline" d="M243 46 L398 314"></path>
			<path class="viz-roc-curve" d="M78 320 L563 40"></path>
			<text class="viz-callout" x="470" y="31">PC1: greatest variance</text>
			<text class="viz-label" x="403" y="316">PC2: discarded</text>
			<path class="viz-operating-guide" d="M438 124 L433 115 M202 236 L207 245 M384 120 L394 138 M256 240 L246 223 M360 169 L355 160 M280 191 L285 200"></path>
			<circle class="viz-node" cx="438" cy="124" r="6"></circle>
			<circle class="viz-node" cx="202" cy="236" r="6"></circle>
			<circle class="viz-node" cx="384" cy="120" r="6"></circle>
			<circle class="viz-node" cx="256" cy="240" r="6"></circle>
			<circle class="viz-node" cx="360" cy="169" r="6"></circle>
			<circle class="viz-node" cx="280" cy="191" r="6"></circle>
			<path class="viz-operating-point" d="M433 109 L439 115 L433 121 L427 115 Z"></path>
			<path class="viz-operating-point" d="M207 239 L213 245 L207 251 L201 245 Z"></path>
			<path class="viz-operating-point" d="M394 132 L400 138 L394 144 L388 138 Z"></path>
			<path class="viz-operating-point" d="M246 217 L252 223 L246 229 L240 223 Z"></path>
			<path class="viz-operating-point" d="M355 154 L361 160 L355 166 L349 160 Z"></path>
			<path class="viz-operating-point" d="M285 194 L291 200 L285 206 L279 200 Z"></path>
			<circle class="viz-operating-point" cx="320" cy="180" r="4"></circle>
			<text class="viz-label" x="326" y="193">mean = 0</text>
			<text class="viz-label" x="350" y="85">perpendicular residual</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> each circle drops perpendicularly to a diamond on PC1. The diamond keeps the coordinate with the largest spread; the dashed segment is the discarded PC2 coordinate and therefore the rank-1 reconstruction error.</figcaption>
</figure>

## Common pitfalls

- **Forgetting to center.** PCA on uncentered data finds the direction toward the mean as PC1, which is rarely what you want.
- **Forgetting to scale.** If features have different units, large-magnitude features dominate; standardize (divide by std) before PCA when units differ.
- **Confusing PCA with whitening.** PCA gives uncorrelated components but not unit variance. Whitening = PCA + scale to unit variance.
- **Using PCA on categorical / sparse data without thought.** PCA assumes Euclidean structure; for sparse / categorical data, look at NMF, LDA, or contrastive embeddings.

## Related

- [Matrices as linear maps](/concepts/matrices-as-linear-maps/). The geometry.
- [Eigenvalues and the spectral theorem](/concepts/eigenvalues-and-spectral-theorem/). For symmetric matrices.
