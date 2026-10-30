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
	<p class="visual-title" id="pd-boundedness-title">A zero-curvature direction lets a quadratic sublevel set extend without bound</p><div class="visual-argument" data-visual-argument="positive-definite-boundedness" data-argument-kind="curve"><p class="visual-argument-premise">A flat direction escapes every finite radius</p><svg viewBox="0 0 360 375" role="img" aria-label="A flat direction escapes every finite radius. For f(x,y)=x squared, the points (0,y) all have loss zero. Positive semidefiniteness alone does not bound sublevel sets."><title>A flat direction escapes every finite radius</title><desc>For f(x,y)=x squared, the points (0,y) all have loss zero. Positive semidefiniteness alone does not bound sublevel sets.</desc><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">-3</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">3</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">9</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-series="Along x: x squared" d="M48 50L52.40625 60.458984375L56.8125 70.5859375L61.21875 80.380859375L65.625 89.84375L70.03125 98.974609375L74.4375 107.7734375L78.84375 116.240234375L83.25 124.375L87.65625 132.177734375L92.0625 139.6484375L96.46875 146.787109375L100.875 153.59375L105.28125 160.068359375L109.6875 166.2109375L114.09375 172.021484375L118.5 177.5L122.90625 182.646484375L127.3125 187.4609375L131.71875 191.943359375L136.125 196.09375L140.53125 199.912109375L144.9375 203.3984375L149.34375 206.552734375L153.75 209.375L158.15625 211.865234375L162.5625 214.0234375L166.96875 215.849609375L171.375 217.34375L175.78125 218.505859375L180.1875 219.3359375L184.59375 219.833984375L189 220L193.40625 219.833984375L197.8125 219.3359375L202.21875 218.505859375L206.625 217.34375L211.03125 215.849609375L215.4375 214.0234375L219.84375 211.865234375L224.25 209.375L228.65625 206.552734375L233.0625 203.3984375L237.46875 199.912109375L241.875 196.09375L246.28125 191.943359375L250.6875 187.4609375L255.09375 182.646484375L259.5 177.5L263.90625 172.021484375L268.3125 166.2109375L272.71875 160.068359375L277.125 153.59375L281.53125 146.787109375L285.9375 139.6484375L290.34375 132.177734375L294.75 124.375L299.15625 116.240234375L303.5625 107.7734375L307.96875 98.974609375L312.375 89.84375L316.78125 80.380859375L321.1875 70.5859375L325.59375 60.458984375L330 50" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></path><path data-proof-series="Along y at x=0: zero" d="M48 220L330 220" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-dasharray:6 3"></path><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">coordinate displacement</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">quadratic value</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Along x: x squared</tspan></text><path d="M20 336L48 336" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="340" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Along y at x=0: zero</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
			<ellipse class="viz-node viz-node--focus" cx="180" cy="144" rx="31" ry="62"></ellipse>
			<path class="viz-operating-guide" d="M149 137V151 M211 137V151 M173 82H187 M173 206H187"></path>
			<text class="viz-label" x="133" y="166">−½</text>
			<text class="viz-label" x="207" y="166">½</text>
			<text class="viz-label" x="190" y="88">1</text>
			<text class="viz-axis-label" x="298" y="137">u</text>
			<text class="viz-axis-label" x="188" y="78">v</text>
			<text class="viz-callout" x="24" y="218">λ = (4, 1) → bounded ellipse</text>
			<rect class="viz-plot-bg" x="12" y="240" width="336" height="228" rx="4"></rect>
			<text class="viz-callout" x="24" y="263">2 · PSD, not PD: one direction has zero cost</text>
			<text class="viz-label" x="24" y="284">A = diag(4, 0) · xᵀAx = 4u² ≤ 1</text>
			<rect class="viz-node viz-node--focus" x="149" y="308" width="62" height="126"></rect>
			<path class="viz-axis" d="M55 372H305 M180 446V296"></path>
			<path class="viz-operating-guide" d="M149 308V434 M211 308V434"></path>
			<path class="viz-pr-curve" d="M180 336V302 M180 406V440" marker-start="url(#pd-arrow-open)" marker-end="url(#pd-arrow-open)"></path>
			<text class="viz-label" x="93" y="389">u = −½</text>
			<text class="viz-label" x="214" y="389">u = ½</text>
			<text class="viz-axis-label" x="298" y="365">u</text>
			<text class="viz-axis-label" x="188" y="304">v</text>
			<text class="viz-callout" x="24" y="459">λ = (4, 0) → v is free → unbounded strip</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> diagonalize first, then inspect one eigendirection at a time. In the top panel, moving along either axis increases the quadratic form, so the threshold closes into an ellipse. In the bottom panel, moving along <var>v</var> adds nothing because its eigenvalue is zero; only <var>u</var> is bounded, and the strip continues forever.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="positive-definite-boundedness"><strong>Check:</strong> Why can v grow without cost for diag(4,0), but not for diag(4,1)?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="positive-definite-boundedness">For f(x,y)=x squared, the points (0,y) all have loss zero. Positive semidefiniteness alone does not bound sublevel sets.</span></figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-25-symmetric-matrices-and-positive-definiteness/">MIT OpenCourseWare 18.06</a> and Boyd and Vandenberghe's <a href="https://web.stanford.edu/~boyd/cvxbook/">Convex Optimization</a>.</p>

## Common pitfalls

- **Calling a non-symmetric matrix PSD.** PSD is defined for symmetric matrices. For asymmetric $A$, the relevant object is $\frac{1}{2}(A + A^\top)$.
- **Trusting numerical eigenvalues at machine precision.** A theoretically PSD covariance computed from data can have tiny negative eigenvalues from rounding. Use jitter ($A + \varepsilon I$) before Cholesky.
- **Confusing PSD with diagonally dominant.** Diagonally dominant with positive diagonal $\Rightarrow$ PSD, but the converse is false.
- **Inverting near-singular PSD matrices.** Always check the smallest eigenvalue or condition number first; regularize if needed.
