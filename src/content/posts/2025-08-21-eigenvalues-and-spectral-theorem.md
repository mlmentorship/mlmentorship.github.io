---
title: "Eigenvalues and the spectral theorem"
description: "Eigenvectors are directions a matrix only stretches. The spectral theorem says symmetric matrices have a full orthogonal eigenbasis with real eigenvalues."
date: "2025-08-21"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A real symmetric matrix acts by independent signed scaling along an orthonormal eigenbasis.
An eigenvector is a nonzero vector $v$ satisfying $Av=\lambda v$; the scalar $\lambda$ is its eigenvalue.
The spectral theorem supplies a complete orthonormal basis, not merely one unchanged direction.
Use this decomposition when a symmetric linear map must be applied, inverted, or understood through its principal directions.

## Eigenvalues, eigenvectors, characteristic polynomial

For square $A\in\mathbb R^{n\times n}$, $(A-\lambda I)v=0$ has a nonzero solution exactly when its determinant vanishes.
The characteristic polynomial has $n$ complex roots counting multiplicity.
Their sum is the trace; their product is the determinant.
Symmetry guarantees real eigenvalues and an orthonormal eigenbasis, including when eigenvalues repeat.

## The spectral theorem (symmetric case)

If $A = A^\top \in \mathbb{R}^{n \times n}$:

$$
A = Q \Lambda Q^\top = \sum_{i=1}^n \lambda_i q_i q_i^\top
$$

where $Q = [q_1 | \dots | q_n]$ is orthogonal ($Q^\top Q = I$) and $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$.

First express the input as $z=Q^\top v$; then scale with $\Lambda$ and return with $Q$.
Negative eigenvalues reverse their axis; zero eigenvalues erase its component.

Take $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$ and $v=(2,1)^\top$.
The diagram uses $q_1=(1,1)^\top/\sqrt2$, $q_2=(-1,1)^\top/\sqrt2$, and eigenvalues $3,1$.
Thus $z=(3,-1)^\top/\sqrt2$, $\Lambda z=(9,-1)^\top/\sqrt2$, and $Av=(5,4)^\top$.

Why does an entire basis exist?
The continuous Rayleigh quotient reaches a maximum on the unit sphere.
Its constrained stationary condition gives $Aq=\lambda q$.
Symmetry makes the orthogonal complement invariant: $q^\top Aw=(Aq)^\top w=0$ whenever $q^\top w=0$.
Repeat on that smaller symmetric subspace to construct the basis.

<!-- visual:spectral-theorem-change-basis -->
<figure class="learning-figure" aria-labelledby="spectral-change-basis-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="spectral-change-basis-title">Change coordinates, scale eigenvector components, then change coordinates back</p><div class="visual-argument" data-visual-argument="spectral-theorem-change-basis" data-argument-kind="ledger" data-argument-composition="geometric-action"><svg viewBox="0 0 360 512" role="img" aria-label="Separate coordinate changes from eigenvalue scaling. For this diagonal example, (3,1) becomes (6,3) under eigenvalues 2 and 3. A general symmetric matrix first uses its orthonormal eigenbasis."><title>Separate coordinate changes from eigenvalue scaling</title><desc>For this diagonal example, (3,1) becomes (6,3) under eigenvalues 2 and 3. A general symmetric matrix first uses its orthonormal eigenbasis.</desc><g opacity=".22"><path d="M34 330L34 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 330L322 330" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M75.14285714285714 330L75.14285714285714 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 288.85714285714283L322 288.85714285714283" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M116.28571428571429 330L116.28571428571429 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 247.71428571428572L322 247.71428571428572" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M157.42857142857144 330L157.42857142857144 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 206.57142857142856L322 206.57142857142856" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M198.57142857142858 330L198.57142857142858 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 165.42857142857142L322 165.42857142857142" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M239.71428571428572 330L239.71428571428572 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 124.28571428571428L322 124.28571428571428" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M280.8571428571429 330L280.8571428571429 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 83.14285714285711L322 83.14285714285711" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M322 330L322 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 42L322 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path></g><path d="M34 330L322 330" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 330L34 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 330L157.42857142857144 288.85714285714283M151.10401610823467 295.1816981774796L157.42857142857144 288.85714285714283L148.57419398009998 287.5922317930755" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><path d="M157.42857142857144 288.85714285714283L280.8571428571429 288.85714285714283M272.8571428571429 292.85714285714283L280.8571428571429 288.85714285714283L272.8571428571429 284.85714285714283" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-linejoin:round"></path><path d="M280.8571428571429 288.85714285714283L280.8571428571429 206.57142857142856M284.8571428571429 214.57142857142856L280.8571428571429 206.57142857142856L276.8571428571429 214.57142857142856" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="157.42857142857144" cy="288.85714285714283" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="280.8571428571429" cy="206.57142857142856" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">In the eigenbasis, each axis scales</tspan><tspan x="20" dy="18">alone</tspan></text><text x="20" y="382" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">First coordinate: 3 x 2 = 6</tspan></text><text x="20" y="424" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Second coordinate: 1 x 3 = 3</tspan></text><text x="20" y="470" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">This diagonal example already uses</tspan><tspan x="20" dy="18">that basis.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-panel">
		<svg viewBox="0 0 360 540" role="img" aria-labelledby="spectral-svg-title spectral-svg-desc">
			<title id="spectral-svg-title">Three-stage action of a symmetric matrix in its eigenbasis</title>
			<desc id="spectral-svg-desc">For the symmetric matrix A with rows two comma one and one comma two, the orthonormal eigenvectors q one and q two have eigenvalues three and one. First, Q transpose expresses v equals two comma one as eigen-coordinates z equals three over square root two comma negative one over square root two. Second, Lambda triples the q one coordinate while leaving the q two coordinate unchanged, producing nine over square root two comma negative one over square root two. Third, Q returns to the original axes and gives A v equals five comma four. Numbered panels, direct labels, solid arrows, and dashed guides convey every step without relying on color.</desc>
			<defs>
				<marker id="spectral-arrow-input" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
				<marker id="spectral-arrow-focus" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="152" rx="4"></rect>
			<text class="viz-callout" x="24" y="34">1 · Change coordinates with Qᵀ</text>
			<text class="viz-label" x="24" y="53">A = [[2, 1], [1, 2]] · v = (2, 1)</text>
			<path class="viz-axis" d="M65 130H275 M170 150V62"></path>
			<path class="viz-baseline" d="M152 148L234 66 M106 66L188 148"></path>
			<text class="viz-axis-label" x="238" y="66">q₁ = (1, 1)/√2</text>
			<text class="viz-axis-label" x="24" y="76">q₂ = (−1, 1)/√2</text>
			<path class="viz-roc-curve" d="M170 130L246 92" marker-end="url(#spectral-arrow-input)"></path>
			<text class="viz-callout" x="253" y="94">v</text>
			<text class="viz-label" x="24" y="153">z = Qᵀv = (3/√2, −1/√2)</text>
			<rect class="viz-plot-bg" x="12" y="178" width="336" height="176" rx="4"></rect>
			<text class="viz-callout" x="24" y="200">2 · Scale independently with Λ = diag(3, 1)</text>
			<path class="viz-axis" d="M58 302H326 M74 334V220"></path>
			<text class="viz-axis-label" x="306" y="332">q₁</text>
			<text class="viz-axis-label" x="48" y="228">q₂</text>
			<path class="viz-operating-guide" d="M74 302V324 M140 302V324 M272 302V324 M140 324H272"></path>
			<path class="viz-roc-curve" d="M74 302L140 324" marker-end="url(#spectral-arrow-input)"></path>
			<circle class="viz-node" cx="140" cy="324" r="4"></circle>
			<text class="viz-label" x="96" y="285">before: z</text>
			<path class="viz-pr-curve" d="M74 302L272 324" marker-end="url(#spectral-arrow-focus)"></path>
			<circle class="viz-operating-point" cx="272" cy="324" r="4"></circle>
			<text class="viz-callout" x="207" y="277">after: Λz</text>
			<text class="viz-label" x="24" y="349">q₁ coordinate ×3 · q₂ coordinate ×1</text>
			<rect class="viz-plot-bg" x="12" y="368" width="336" height="160" rx="4"></rect>
			<text class="viz-callout" x="24" y="390">3 · Return to the original axes with Q</text>
			<path class="viz-axis" d="M56 498H318 M94 516V400"></path>
			<path class="viz-baseline" d="M94 498L142 474"></path>
			<text class="viz-label" x="147" y="477">v = (2, 1)</text>
			<path class="viz-pr-curve" d="M94 498L214 402" marker-end="url(#spectral-arrow-focus)"></path>
			<text class="viz-callout" x="222" y="414">Av = (5, 4)</text>
			<text class="viz-label" x="24" y="521">Q(Λz) = QΛQᵀv = (5, 4)</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> <var>Q</var><sup>T</sup> changes coordinates without changing length. <var>Λ</var> triples the first coordinate and preserves the second. <var>Q</var> returns the result to the original axes: <var>Av</var> = (5, 4).</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="spectral-theorem-change-basis"><strong>Check:</strong> Which step changes only coordinates? Which step actually scales the eigenvector components?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="spectral-theorem-change-basis">For this diagonal example, (3,1) becomes (6,3) under eigenvalues 2 and 3. A general symmetric matrix first uses its orthonormal eigenbasis. <br><strong>Does scaling the first eigen-coordinate require mixing it with the second?</strong></span></figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-25-symmetric-matrices-and-positive-definiteness/">MIT OpenCourseWare 18.06</a> and Sheldon Axler's open-access <a href="https://linear.axler.net/">Linear Algebra Done Right</a>.</p>

## Connection to SVD

For symmetric positive-semidefinite $A$, one SVD uses $U=V=Q$ and $\Sigma=\Lambda$.
For symmetric indefinite $A$, singular values are $|\lambda_i|$, not the signed eigenvalues.
General rectangular matrices have an SVD but no square-matrix eigendecomposition of the same form.

## Change the Requirement

### Apply the Map Repeatedly

**New task:** Compute $A^2v$ without two dense applications after the eigenbasis is available.

**Picture and change:** Replace each $\lambda_i$ by $\lambda_i^2$, preserving the coordinate basis.

**Check:** $Q\operatorname{diag}(9,1)Q^\top v=(14,13)^\top$, matching $A(5,4)^\top$.

**Use when:** Many powers or repeated queries reuse the same decomposition.
Dense decomposition costs $O(n^3)$; one dense basis transform costs $O(n^2)$.

### Solve Instead of Apply

**New task:** Recover $x$ satisfying $Ax=v$.

**Picture and change:** Divide each eigen-coordinate by its nonzero eigenvalue.

**Check:** $x=Q\operatorname{diag}(1/3,1)Q^\top(2,1)^\top=(1,0)^\top$.

**Use when:** Every eigenvalue is nonzero.
A zero eigenvalue makes some right-hand sides impossible and others nonunique.
Adding a ridge changes the system; it is not an exact inverse of the original map.

<!-- learning:core-end -->
<div data-learning-extension>

## Where eigenvalues show up in ML

| Object | What its eigenvalues tell you |
|--------|------------------------------|
| Covariance matrix | Variances along principal axes (PCA) |
| Positive-definite Hessian | Local curvature; spectral condition number $\lambda_{\max}/\lambda_{\min}$ |
| Graph Laplacian | Connectivity, spectral clustering, GNN smoothness |
| Reversible Markov chain | Mixing depends on nontrivial eigenvalue magnitudes and the chain's assumptions |
| General attention score matrix | May be asymmetric; singular values describe rank and norm structure |
| Recurrent weight matrix | Long-run dynamics; finite-step gradient norms also depend on singular values and nonnormality |

## Common pitfalls

- **Dropping symmetry.** The matrix $\begin{bmatrix}1&1\\0&1\end{bmatrix}$ has only one independent eigenvector.
	Its repeated eigenvalue does not provide a basis.
- **Dropping positivity.** $\operatorname{diag}(-2,1)$ has eigenvalues $-2,1$ but singular values $2,1$.
- **Ignoring scale.** Dense decompositions need cubic arithmetic and quadratic storage.
	Lanczos targets symmetric eigenproblems; Arnoldi handles more general operators.
	Approximate methods require residual and convergence checks.

## Reconstruct

Why may repeated eigenvalues still have orthogonal eigenvectors for a symmetric matrix?

<details>
<summary>Reasoned answer</summary>

Inside a repeated-eigenvalue eigenspace, every linear combination remains an eigenvector with that eigenvalue.
Choose an orthonormal basis inside that subspace.
Symmetry, not distinctness of all eigenvalues, guarantees a complete orthonormal collection.

</details>

## Related

- [SVD and PCA](/concepts/svd-and-pca/). Generalization to all matrices.
- [Positive definite matrices](/concepts/positive-definite-matrices/). The cone of PSD matrices.

</div>
