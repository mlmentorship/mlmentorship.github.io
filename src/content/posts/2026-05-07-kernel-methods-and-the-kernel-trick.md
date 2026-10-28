---
title: "Kernel methods and the kernel trick"
description: "Compute inner products in a high-dimensional feature space without ever materializing the features. The mathematical move that lets a linear classifier draw nonlinear boundaries."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A kernel computes a feature-space inner product without explicitly constructing the feature vectors.
Write $k(x,x')=\langle\phi(x),\phi(x')\rangle$.
An algorithm expressed through these inner products can use the kernel directly, preserving the feature-space calculation.
The shortcut avoids feature construction; it does not automatically avoid pairwise dataset storage or expensive training solves.

## The trick

Suppose the desired feature-space algorithm reads inputs only through $\langle\phi(x_i),\phi(x_j)\rangle$.
For two-dimensional inputs, consider

$$
k(x, x') = (x^\top x' + 1)^2.
$$

An exact feature map is

$$
\phi(x)=(x_1^2,\sqrt2x_1x_2,x_2^2,\sqrt2x_1,\sqrt2x_2,1).
$$

Expanding the feature inner product reproduces the squared polynomial, including its cross-term coefficients.
For $x=(2,1)$ and $x'=(1,3)$, the six contributions are $4,12,9,4,6,1$, totaling $36$.
The shortcut obtains $(5+1)^2=36$ from the original coordinates.
In $d$ dimensions, explicit degree-two features number $\binom{d+2}{2}$; one kernel evaluation needs $O(d)$ arithmetic.

<!-- visual:kernel-trick-exact-shortcut -->
<figure class="learning-figure plot-panel" aria-labelledby="kernel-shortcut-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="kernel-shortcut-title">Which work does the kernel trick skip?</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="kernel-shortcut-svg-title kernel-shortcut-svg-desc">
		<title id="kernel-shortcut-svg-title">Two exact routes to the same degree-two polynomial-kernel value</title>
		<desc id="kernel-shortcut-svg-desc">For x equals 2 comma 1 and x prime equals 1 comma 3, the explicit route constructs two six-dimensional feature vectors and dots them to get 4 plus 12 plus 9 plus 4 plus 6 plus 1 equals 36. The kernel shortcut stays in two dimensions: x dot x prime equals 5, so open parenthesis 5 plus 1 close parenthesis squared also equals 36. The shortcut returns exactly the same scalar without materializing either feature vector.</desc>
		<defs>
			<marker id="kernel-shortcut-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-edge)"></path></marker>
			<marker id="kernel-shortcut-focus-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-focus-stroke)"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="8" width="344" height="233" rx="5"></rect>
		<text class="viz-axis-label" x="20" y="31">EXPLICIT FEATURE ROUTE · EXACT, BUT MORE WORK</text>
		<rect class="viz-node viz-node--input" x="73" y="44" width="214" height="36" rx="4"></rect>
		<text class="viz-callout" x="180" y="67" text-anchor="middle">x = (2, 1) · x′ = (1, 3)</text>
		<path d="M180 80V98" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#kernel-shortcut-arrow)"></path>
		<rect class="viz-node" x="28" y="101" width="304" height="66" rx="4"></rect>
		<text class="viz-axis-label" x="180" y="120" text-anchor="middle">MATERIALIZE BOTH 6-D VECTORS</text>
		<text class="viz-label" x="180" y="140" text-anchor="middle">φ(x) = [4, 2√2, 1, 2√2, √2, 1]</text>
		<text class="viz-label" x="180" y="157" text-anchor="middle">φ(x′) = [1, 3√2, 9, √2, 3√2, 1]</text>
		<path d="M180 167V184" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#kernel-shortcut-arrow)"></path>
		<rect class="viz-node viz-node--output" x="51" y="187" width="258" height="39" rx="4"></rect>
		<text class="viz-callout" x="180" y="203" text-anchor="middle">φ(x) · φ(x′)</text>
		<text class="viz-node-value" x="180" y="219">4 + 12 + 9 + 4 + 6 + 1 = 36</text>
		<rect class="viz-plot-bg" x="8" y="253" width="344" height="169" rx="5"></rect>
		<text class="viz-axis-label" x="20" y="276">KERNEL ROUTE · SAME SCALAR, NO FEATURE VECTORS</text>
		<rect class="viz-node viz-node--input" x="25" y="291" width="122" height="46" rx="4"></rect>
		<text class="viz-callout" x="86" y="309" text-anchor="middle">stay in R²</text>
		<text class="viz-node-value" x="86" y="327">x · x′ = 5</text>
		<path d="M147 314H176" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-dasharray:5 3;marker-end:url(#kernel-shortcut-focus-arrow)"></path>
		<text class="viz-label" x="162" y="303" text-anchor="middle">skip φ</text>
		<rect class="viz-node viz-node--focus" x="179" y="291" width="156" height="46" rx="4"></rect>
		<text class="viz-callout" x="257" y="309" text-anchor="middle">k(x, x′)</text>
		<text class="viz-node-value" x="257" y="327">(5 + 1)² = 36</text>
		<path d="M257 337V357" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;marker-end:url(#kernel-shortcut-focus-arrow)"></path>
		<rect class="viz-node viz-node--output" x="77" y="360" width="206" height="42" rx="4"></rect>
		<text class="viz-callout" x="180" y="378" text-anchor="middle">same inner product: 36</text>
		<text class="viz-node-value" x="180" y="395">exact shortcut, not an approximation</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> both routes return 36. The explicit route constructs six-coordinate features; the kernel route evaluates the polynomial directly. This skips feature construction without approximating the inner product. References: <a href="https://link.springer.com/article/10.1023/A:1009715923555">Burges's SVM tutorial</a> and <a href="https://scikit-learn.org/stable/modules/svm.html#svm-kernels">scikit-learn's SVM formulation</a>. <span class="visual-reader-check" data-reader-check="kernel-trick-exact-shortcut"><strong>Check:</strong> Which calculation gives the same 36 without materializing either six-dimensional feature vector?</span></figcaption>
</figure>

### Change the Distance Scale

**New task:** Define similarity through distance rather than a fixed polynomial expansion.

**Picture and change:** Use the Gaussian RBF kernel $k(x,x')=\exp(-\lVert x-x'\rVert^2/(2\sigma^2))$, with $\sigma>0$.

**Check:** The example's squared distance is $5$.
Bandwidths $\sigma=1,2$ give $e^{-5/2}\approx0.0821$ and $e^{-5/8}\approx0.5353$.

**Use when:** A distance-based smooth similarity matches the task.
Over a continuous input domain, this kernel generally needs an infinite-dimensional exact feature space, not a finite polynomial map.

## The Gram matrix

For $N$ points, the Gram matrix has entries $K_{ij}=k(x_i,x_j)$ and shape $N\times N$.

- **Kernel ridge regression**: solve $(K+\lambda I)\alpha=y$, then predict $\sum_i\alpha_i k(x_i,x)$.
	This convention uses an unnormalized squared-error sum plus $\lambda$ times the squared function norm.
- **Kernel PCA**: eigendecompose the centered $K$.
- **Gaussian processes**: use $K + \sigma^2 I$ as the covariance.
- **SVM**: dual formulation depends only on $K$ and labels.

**New task:** Train when a dense Gram matrix exceeds the memory budget.

**Picture and change:** Approximate $K$ with landmark columns $C$ and their Gram matrix $W$: $\widetilde K=CW^\dagger C^\top$.

**Check:** For $K=I_2$ and only the first landmark, $\widetilde K=\operatorname{diag}(1,0)$.
The second direction is lost; this is no longer the exact kernel trick.

**Use when:** Measured approximation error is acceptable.
A float64 matrix with $100{,}000$ rows requires $80$ decimal GB before factorization or solver workspace.
Dense direct solves cost cubic arithmetic; practical limits depend on hardware and solver structure.

## What makes a valid kernel

A real kernel must be symmetric and give a positive-semidefinite Gram matrix for every finite input collection.
Feature inner products satisfy this because

$$
a^\top Ka=\left\lVert\sum_i a_i\phi(x_i)\right\rVert^2\ge0.
$$

The converse supplies a Hilbert-space feature representation.
Mercer expansions additionally require their own domain and regularity conditions.

Common kernels:

- **Linear**: $k(x, x') = x^\top x'$. The trivial case.
- **Polynomial**: $(x^\top x'+c)^p$, with nonnegative integer degree $p$ and $c\ge0$.
- **RBF / Gaussian**: $\exp(-\gamma\lVert x-x'\rVert^2)$, with $\gamma\ge0$.
- **Laplacian**: $\exp(-\gamma\lVert x-x'\rVert_1)$, with $\gamma\ge0$.
- **String kernels**: count shared substrings.
- **Graph kernels**: count shared subgraphs.

Sums, pointwise products, and nonnegative scalings preserve validity; arbitrary negative scaling does not.

<!-- learning:core-end -->
<div data-learning-extension>

## Kernel trick in attention

An inner product in a common feature space is a linear kernel.
Attention may use different query and key maps, producing an asymmetric cross-score matrix.
Row-normalized attention is not automatically a symmetric PSD Gram matrix.
Feature-map attention methods require their particular approximation and normalization guarantees; they are not all RBF random-feature substitutions.

## Why deep learning won

Learned representations can adapt similarity to a task; a fixed kernel imposes a chosen geometry.
Neither guarantees better results across all datasets.
Kernels remain useful for structured prior similarity, Gaussian processes, controlled baselines, and theoretical analysis.

## Common pitfalls

- **Missing coefficients.** Omitting the square-root-two factors changes the polynomial feature inner product.
- **Uncontrolled scale.** Feature units change RBF distances and the effective bandwidth.
- **Confusing shortcuts.** An exact kernel evaluation and an approximate low-rank Gram matrix make different guarantees.
- **Overloaded terminology.** A convolution filter called a kernel need not define a PSD similarity function.

## Reconstruct

<details>
<summary>Does a symmetric similarity matrix always define a valid kernel?</summary>

No. $\begin{bmatrix}1&2\\2&1\end{bmatrix}$ is symmetric but has eigenvalue $-1$.
For $a=(1,-1)^\top$, its quadratic form is $-2$, which cannot be a squared feature norm.

</details>

## Related

- [SVM and the kernel trick](/concepts/svm-and-kernels/).
- [Gaussian processes](/concepts/gaussian-processes/).
- [The attention mechanism](/concepts/attention-mechanism/).

</div>
