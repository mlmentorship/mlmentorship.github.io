---
title: "Determinant and volume"
description: "The determinant of a matrix is the signed volume scaling factor of the linear map. Zero determinant means the map collapses dimensions."
date: "2025-12-02"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

The determinant $\det(A)$ of a square matrix $A$ is the signed factor by which $A$ scales $n$-dimensional volumes. Geometrically, $|\det(A)|$ is the volume of the parallelepiped spanned by $A$'s column vectors; the sign flips if $A$ contains a reflection.

Determinants tell you whether a linear map is invertible (non-zero det) or singular (zero det). They appear in change-of-variables formulas (probability density transformations, normalizing flows), in computing volumes of parallelotopes (Gaussian likelihoods), and as Jacobian determinants in differential geometry.

## Properties

For $A, B \in \mathbb{R}^{n \times n}$:

- $\det(AB) = \det(A) \det(B)$. Composition multiplies volumes.
- $\det(A^\top) = \det(A)$.
- $\det(A^{-1}) = 1/\det(A)$ when $A$ is invertible.
- $\det(cA) = c^n \det(A)$ for scalar $c$.
- $\det = 0 \iff$ columns linearly dependent $\iff$ $A$ not invertible $\iff$ $A$ has a zero eigenvalue.
- $\det(A) = \prod_i \lambda_i$ (product of eigenvalues, with multiplicity, possibly complex).
- For triangular matrices: $\det = \prod_i A_{ii}$ (product of diagonal).

## Geometric interpretation

The columns of $A$ are vectors in $\mathbb{R}^n$. They span a parallelepiped. Its volume is $|\det(A)|$.

- $\det(A) > 0$: $A$ preserves orientation.
- $\det(A) < 0$: $A$ reverses orientation (contains a reflection).
- $\det(A) = 0$: parallelepiped is flat. Columns are linearly dependent. $A$ collapses at least one dimension.

For an orthogonal matrix $Q$: $|\det(Q)| = 1$ (rotation/reflection preserves volume).

<!-- visual:determinant-signed-area-collapse -->
<figure class="learning-figure" aria-labelledby="determinant-area-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="determinant-area-title">Separate area magnitude from orientation sign, then see why dependent columns have zero area.</p>
	<div class="visual-panel">
		<svg viewBox="0 0 360 560" role="img" aria-labelledby="determinant-area-svg-title determinant-area-svg-desc">
			<title id="determinant-area-svg-title">Three ordered column pairs with positive, negative, and zero determinant</title>
			<desc id="determinant-area-svg-desc">The first panel shows columns a one equals two comma one half and a two equals one half comma one and one quarter, spanning a parallelogram of area two and one quarter in counterclockwise order, so the determinant is positive two and one quarter. The second panel uses the exact same parallelogram but swaps the column order: a one now points to one half comma one and one quarter and a two points to two comma one half. Its area remains two and one quarter, but the order is clockwise, so the determinant is negative two and one quarter. The third panel has columns b one equals two comma one and b two equals one comma one half. The second is half the first, so both arrows lie on one line, the parallelogram collapses to a segment, and the determinant is zero.</desc>
			<defs>
				<marker id="det-arrow-primary" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
				<marker id="det-arrow-secondary" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="164" rx="4"></rect>
			<text class="viz-callout" x="24" y="34">1 · Positive: area 2.25, orientation preserved</text>
			<path class="viz-gridline" d="M45 148H225 M65 160V48"></path>
			<polygon class="viz-node viz-node--focus" points="65,148 145,128 165,78 85,98"></polygon>
			<path d="M65 148L145 128" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 148L85 98" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="148" r="3"></circle>
			<text class="viz-label" x="139" y="146">a₁ = (2, 0.5)</text>
			<text class="viz-label" x="91" y="92">a₂ = (0.5, 1.25)</text>
			<text class="viz-callout" x="238" y="83">counterclockwise</text>
			<text class="viz-node-value" x="290" y="104">det[a₁ a₂]</text>
			<text class="viz-node-label" x="290" y="129">= +2.25</text>
			<rect class="viz-plot-bg" x="12" y="190" width="336" height="164" rx="4"></rect>
			<text class="viz-callout" x="24" y="212">2 · Swap columns: same area, opposite sign</text>
			<path class="viz-gridline" d="M45 326H225 M65 338V226"></path>
			<polygon class="viz-node viz-node--focus" points="65,326 145,306 165,256 85,276"></polygon>
			<path d="M65 326L85 276" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 326L145 306" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="326" r="3"></circle>
			<text class="viz-label" x="91" y="270">a₁ = (0.5, 1.25)</text>
			<text class="viz-label" x="139" y="324">a₂ = (2, 0.5)</text>
			<text class="viz-callout" x="238" y="261">clockwise</text>
			<text class="viz-node-value" x="290" y="282">det[a₁ a₂]</text>
			<text class="viz-node-label" x="290" y="307">= −2.25</text>
			<rect class="viz-plot-bg" x="12" y="368" width="336" height="180" rx="4"></rect>
			<text class="viz-callout" x="24" y="390">3 · Dependent columns: one direction is lost</text>
			<path class="viz-gridline" d="M45 522H225 M65 534V414"></path>
			<path d="M65 522L145 482" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 522L105 502" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="522" r="3"></circle>
			<text class="viz-label" x="139" y="476">b₁ = (2, 1)</text>
			<text class="viz-label" x="107" y="516">b₂ = (1, 0.5) = ½b₁</text>
			<text class="viz-callout" x="238" y="447">collapsed segment</text>
			<text class="viz-node-value" x="290" y="468">det[b₁ b₂]</text>
			<text class="viz-node-label" x="290" y="493">= 0</text>
			<text class="viz-label" x="238" y="519">area = 0 · not invertible</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the first two panels before looking at the third. Swapping the ordered columns leaves the parallelogram and its area unchanged but reverses orientation, so only the sign flips. When one column becomes a multiple of the other, the shape loses a dimension; its area and determinant both become zero.</figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-18-properties-of-determinants/">MIT OpenCourseWare 18.06</a>, the <a href="https://www.deeplearningbook.org/contents/linear_algebra.html">Deep Learning linear algebra chapter</a>, and the open <a href="https://math.libretexts.org/Bookshelves/Linear_Algebra/Interactive_Linear_Algebra_(Margalit_and_Rabinoff)/04%3A_Determinants/4.01%3A_Determinants-_Definition">Interactive Linear Algebra determinant text</a>.</p>

## Change of variables (probability)

If $Y = f(X)$ with $f$ invertible and differentiable, the density of $Y$ is

$$
p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)|
$$

where $J_{f^{-1}}$ is the Jacobian of the inverse transform. This is the basis of normalizing flows: pick $f$ so the Jacobian determinant is cheap to compute (triangular Jacobian → $\det = \prod_i \partial y_i / \partial x_i$).

## Computing determinants

| Method | Cost | When |
|--------|------|------|
| LU decomposition | $O(n^3)$ | General-purpose; standard library default |
| Triangular: product of diag | $O(n)$ | When matrix is already triangular |
| Eigendecomposition: $\prod \lambda_i$ | $O(n^3)$ | If you need eigenvalues anyway |
| **Log-determinant** for PD matrices | via Cholesky | When you only need $\log \det$ (e.g., Gaussian log-likelihood) |

Numerical tip: for a general matrix, use a signed log-determinant: track the row-pivot sign and the signs of the LU diagonal entries, then sum $\log |U_{ii}|$. For a positive-definite matrix with $A = LL^\top$, compute $\log \det(A) = 2\sum_i \log L_{ii}$. The determinant itself overflows or underflows quickly.

## Common pitfalls

- **Using $\det$ as a proxy for matrix "size."** A nearly singular matrix can have huge entries but tiny $\det$.
- **Computing $\det$ for invertibility tests.** Numerically unstable; use rank or condition number instead.
- **Forgetting the absolute value in change-of-variables formulas.** Densities are non-negative; the Jacobian determinant can be negative.
