---
title: "SVD and PCA"
description: "Separate singular directions of a linear map from principal directions of centered data, then choose rank using reconstruction error."
date: "2025-09-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

SVD identifies input directions, output directions, and nonnegative scaling factors for any real matrix.
PCA applies that decomposition to centered observations, retaining directions that minimize Euclidean squared reconstruction error.
Large retained variance does not guarantee retained label information.

## The decomposition

$A\in\mathbb R^{m\times n}$ has $A=U\Sigma V^\top$.
Full factors have orthogonal $U:m\times m$, orthogonal $V:n\times n$, and rectangular diagonal $\Sigma:m\times n$.

- The first $r$ columns of $V$ span the row space; remaining columns span the nullspace.
- The first $r$ columns of $U$ span the column space; remaining columns span the left nullspace.
- Singular values satisfy $\sigma_1\ge\sigma_2\ge\cdots\ge0$; rank $r$ counts strictly positive values in exact arithmetic.

The orthogonal factors can include reflections, not just rotations.
For an invertible square map, the spectral condition number is $\sigma_1/\sigma_n$.
Null directions prevent unrestricted inversion; $\sigma_1/\sigma_r$ describes conditioning only on the retained nonzero subspace.

## Truncated SVD and low-rank approximation

For rank at most $k$, a best approximation in Frobenius or spectral norm is

$$
A_k = U_k \Sigma_k V_k^\top
$$

Keep the leading $k$ singular values and their corresponding directions.
Orthogonal changes of basis preserve these norms, reducing the error calculation to discarded diagonal entries:

$$
\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2,\qquad
\|A-A_k\|_2=\sigma_{k+1}.
$$

Take omitted singular values as zero when none remain.
This is the Eckart–Young result; a low-rank model constraint does not automatically prescribe truncating an unrelated matrix.

## PCA as SVD

For $n>1$ observations in rows, fit a training mean $\mu$ and center $\widetilde X=X-\mathbf1\mu^\top$.

1. Decompose $\widetilde X=U\Sigma V^\top$.
2. Use leading columns $V_k$ as feature-space principal directions.
3. Compute scores $Z=\widetilde XV_k$ and reconstruct $\widehat X=ZV_k^\top+\mathbf1\mu^\top$.
4. Component sample variance is $\sigma_i^2/(n-1)$.

The covariance is $\widetilde X^\top\widetilde X/(n-1)$; its eigenvectors are these same principal directions.
Direct SVD avoids explicitly forming a Gram matrix, which squares the nonzero singular-value condition ratio.

For an exact numerical companion to the schematic, use rows $(-2,-1),(-2,1),(2,-1),(2,1)$.
They are centered, and $X^\top X=\operatorname{diag}(16,4)$, giving singular values $4,2$ and $V=I$.
Rank-one scores are $(-2,-2,2,2)^\top$; reconstruct each second coordinate as zero.
Four unit residuals give squared error $4$, while retained variance is $16/(16+4)=80\%$.

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
	<figcaption><strong>Read it this way:</strong> each circle projects perpendicularly to a diamond on PC1. The diamond retains its PC1 coordinate. The dashed segment is the discarded residual. This six-point schematic illustrates the projection, not the separate four-point arithmetic fixture.</figcaption>
</figure>

## Change the Requirement

### Retain Ninety Percent of Variance

**New task:** Choose the smallest rank retaining at least $90\%$ of total squared singular-value mass.

**Picture and change:** Accumulate $\sigma_i^2$ in descending order; change scores and reconstruction dimensions together.

**Check:** The fixture's first component retains only $80\%$; rank two retains $100\%$ and reconstructs exactly.

**Use when:** A declared reconstruction requirement determines storage or representation size.

### Standardize the Features

**New task:** Give both features equal sample variance before fitting PCA.

**Picture and change:** Divide centered columns by their training sample standard deviations, changing the feature-space metric.

**Check:** Fixture covariance becomes $I$; every unit direction is a possible first component, retaining $50\%$.

**Use when:** Units or domain priorities justify equalized feature weighting.
Carry the fitted mean, scales, and directions into inference; refitting on test data changes the transformation.

<!-- learning:core-end -->
<div data-learning-extension>

## Common pitfalls

- **Different objective.** Uncentered SVD includes the mean's contribution; its leading direction need not equal the mean direction.
- **Automatic scaling.** Standardization changes the objective and is undefined for zero-variance columns without a policy.
- **Whitening confusion.** PCA scores are uncorrelated, not generally unit variance.
  Dividing by positive component standard deviations whitens them; zero singular values need exclusion or regularization.
- **Direction ambiguity.** Singular-vector signs and tied-eigenvalue bases can vary without changing the reconstructed subspace.
- **Data geometry.** Centering sparse matrices can destroy sparsity; categorical data need a justified metric.

## Reconstruct

<details>
<summary>What if the label is the sign of the second feature?</summary>

The rank-one fixture discards that feature completely despite retaining eighty percent of variance.
Unsupervised reconstruction quality is not a guarantee of predictive sufficiency.

</details>

</div>

## Related

- [Matrices as linear maps](/concepts/matrices-as-linear-maps/). The geometry.
- [Eigenvalues and the spectral theorem](/concepts/eigenvalues-and-spectral-theorem/). For symmetric matrices.
