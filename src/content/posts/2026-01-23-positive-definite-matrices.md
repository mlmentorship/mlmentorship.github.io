---
title: "Positive (semi-)definite matrices"
description: "Matrices that define inner products and proper covariances. The geometry of PSD: ellipsoids, not arbitrary shapes."
date: "2026-01-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is **positive semi-definite (PSD)** if $x^\top A x \ge 0$ for all $x \in \mathbb{R}^n$, and **positive definite (PD)** if $x^\top A x > 0$ for all $x \ne 0$. Equivalently: all eigenvalues are non-negative (PSD) or strictly positive (PD).

PSD matrices are the matrices that can serve as **covariance matrices**, **kernel matrices** (Gram matrices), **inner-product weight matrices**, and **Hessians at local minima**. The PSD cone is the natural domain for many optimization problems (semidefinite programming, Gaussian processes, kernel methods).

## Equivalent characterizations

For symmetric $A$:

- $x^\top A x \ge 0$ for all $x$ (definition).
- All eigenvalues $\lambda_i \ge 0$.
- $A = B^\top B$ for some matrix $B$ (factorization, e.g., Cholesky $B = L^\top$ with $L$ lower triangular).
- $A$ is the covariance matrix of some random vector.
- All principal minors are non-negative. For PD matrices, Sylvester's criterion gives the simpler equivalent test that every leading (upper-left) principal minor is positive.

For PD: replace non-negative eigenvalues and quadratic forms with strictly positive ones, and require $B$ to be invertible in the factorization.

## The Cholesky factorization

Every PD matrix has a unique decomposition $A = L L^\top$ with $L$ lower triangular and positive diagonal. This is the standard way to:

- Solve $Ax = b$ when $A$ is PD ($O(n^3/3)$ instead of $O(n^3)$ for general LU).
- Sample from a Gaussian: if $z \sim \mathcal{N}(0, I)$ then $L z \sim \mathcal{N}(0, A)$.
- Compute Gaussian log-likelihoods: $\log \det A = 2 \sum_i \log L_{ii}$.

PSD (not strictly PD) matrices admit Cholesky-like decompositions but with possible zero diagonal entries; use pivoted Cholesky or LDL.

## The PSD cone

The set of $n \times n$ PSD matrices forms a **convex cone** (closed under non-negative combinations). This is why semidefinite programming generalizes linear programming. It optimizes over a different cone.

Operations preserving PSD:

- $A + B$ is PSD if $A, B$ are PSD.
- $c A$ is PSD for $c \ge 0$.
- $B^\top A B$ is PSD for any compatible $B$.
- Element-wise (Hadamard) product (Schur product theorem).

Operations *not* preserving PSD:

- General matrix product $AB$ (only if $A, B$ commute).
- Inverse: PD matrices have PD inverses; PSD with zero eigenvalue is not invertible.

## Geometric intuition

For $A$ PD, the set $\{x : x^\top A x \le 1\}$ is a closed ellipsoid centered at the origin. Eigenvectors of $A$ give the axes; eigenvalues give $1/\text{axis-length}^2$. If $A$ is PSD but singular, a zero eigenvalue leaves its eigenvector direction unpenalized, so the same sublevel set is an **unbounded cylinder** (a strip in 2D), not a flat ellipsoid. A singular covariance distribution is instead supported on a lower-dimensional subspace; that is the setting in which its probability contours collapse.

<!-- visual:positive-definite-boundedness -->
<figure class="learning-figure" aria-labelledby="pd-boundedness-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="pd-boundedness-title">Predict whether a quadratic sublevel set is bounded by checking whether every eigenvalue penalizes its eigenvector direction.</p>
	<div class="visual-panel">
		<svg viewBox="0 0 360 480" role="img" aria-labelledby="pd-boundedness-svg-title pd-boundedness-svg-desc">
			<title id="pd-boundedness-svg-title">Positive definite ellipse compared with a singular positive semidefinite strip</title>
			<desc id="pd-boundedness-svg-desc">Two numbered coordinate plots use the same threshold x transpose A x less than or equal to one. In the first, A equals diagonal four comma one, so the quadratic form is four u squared plus v squared. Both eigenvalues are positive, and the feasible set is a bounded ellipse with u semiaxis one half and v semiaxis one. In the second, A equals diagonal four comma zero, so the form is four u squared. The v coordinate has zero cost, and the feasible set is the unbounded vertical strip from u equals negative one half to positive one half. Direct equations, boundary labels, solid axes, dashed strip edges, and continuation arrows communicate the distinction without color.</desc>
			<defs>
				<marker id="pd-arrow-open" markerWidth="7" markerHeight="7" refX="3.5" refY="1" orient="auto"><path class="viz-arrow-forward" d="M0 7L3.5 0L7 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="214" rx="4"></rect>
			<text class="viz-callout" x="24" y="35">1 · PD: every direction has positive cost</text>
			<text class="viz-label" x="24" y="56">A = diag(4, 1) · xᵀAx = 4u² + v² ≤ 1</text>
			<path class="viz-axis" d="M55 144H305 M180 210V72"></path>
			<ellipse class="viz-node viz-node--focus" cx="180" cy="144" rx="48" ry="62"></ellipse>
			<path class="viz-operating-guide" d="M132 137V151 M228 137V151 M173 82H187 M173 206H187"></path>
			<text class="viz-label" x="116" y="166">−½</text>
			<text class="viz-label" x="224" y="166">½</text>
			<text class="viz-label" x="190" y="88">1</text>
			<text class="viz-axis-label" x="298" y="137">u</text>
			<text class="viz-axis-label" x="188" y="78">v</text>
			<text class="viz-callout" x="24" y="218">λ = (4, 1) → bounded ellipse</text>
			<rect class="viz-plot-bg" x="12" y="240" width="336" height="228" rx="4"></rect>
			<text class="viz-callout" x="24" y="263">2 · PSD, not PD: one direction has zero cost</text>
			<text class="viz-label" x="24" y="284">A = diag(4, 0) · xᵀAx = 4u² ≤ 1</text>
			<rect class="viz-node viz-node--focus" x="132" y="308" width="96" height="126"></rect>
			<path class="viz-axis" d="M55 372H305 M180 446V296"></path>
			<path class="viz-operating-guide" d="M132 308V434 M228 308V434"></path>
			<path class="viz-pr-curve" d="M180 336V302 M180 406V440" marker-start="url(#pd-arrow-open)" marker-end="url(#pd-arrow-open)"></path>
			<text class="viz-label" x="104" y="389">u = −½</text>
			<text class="viz-label" x="230" y="389">u = ½</text>
			<text class="viz-axis-label" x="298" y="365">u</text>
			<text class="viz-axis-label" x="188" y="304">v</text>
			<text class="viz-callout" x="24" y="459">λ = (4, 0) → v is free → unbounded strip</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> diagonalize first, then inspect one eigendirection at a time. In the top panel, moving along either axis increases the quadratic form, so the threshold closes into an ellipse. In the bottom panel, moving along <var>v</var> adds nothing because its eigenvalue is zero; only <var>u</var> is bounded, and the strip continues forever.</figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-25-symmetric-matrices-and-positive-definiteness/">MIT OpenCourseWare 18.06</a> and Boyd and Vandenberghe's <a href="https://web.stanford.edu/~boyd/cvxbook/">Convex Optimization</a>.</p>

## Common pitfalls

- **Calling a non-symmetric matrix PSD.** PSD is defined for symmetric matrices. For asymmetric $A$, the relevant object is $\frac{1}{2}(A + A^\top)$.
- **Trusting numerical eigenvalues at machine precision.** A theoretically PSD covariance computed from data can have tiny negative eigenvalues from rounding. Use jitter ($A + \varepsilon I$) before Cholesky.
- **Confusing PSD with diagonally dominant.** Diagonally dominant with positive diagonal $\Rightarrow$ PSD, but the converse is false.
- **Inverting near-singular PSD matrices.** Always check the smallest eigenvalue or condition number first; regularize if needed.
