---
title: "Eigenvalues and the spectral theorem"
description: "Eigenvectors are directions a matrix only stretches. The spectral theorem says symmetric matrices have a full orthogonal eigenbasis with real eigenvalues."
date: "2025-08-21"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An **eigenvector** of $A$ is a non-zero vector $v$ such that $A v = \lambda v$ for some scalar $\lambda$ (the **eigenvalue**). $A$ acts on $v$ purely by scaling. The **spectral theorem** states that every real symmetric matrix is orthogonally diagonalizable: $A = Q \Lambda Q^\top$ with $Q$ orthogonal and $\Lambda$ diagonal real.

Eigendecompositions explain stability of dynamical systems, convergence of optimization, structure of covariance matrices, and properties of attention / graph operators. The spectral theorem is the mathematical reason PCA works on covariance matrices, why Laplacian eigenmaps make sense for graphs, and why second-order optimizers reason about Hessian eigenvalues.

## Eigenvalues, eigenvectors, characteristic polynomial

For a square matrix $A \in \mathbb{R}^{n \times n}$:

- $A v = \lambda v$ is equivalent to $(A - \lambda I) v = 0$, so eigenvalues are roots of $\det(A - \lambda I) = 0$ (the characteristic polynomial).
- An $n \times n$ matrix has $n$ eigenvalues (counted with multiplicity), possibly complex, possibly repeated.
- Trace $= \sum_i \lambda_i$. Determinant $= \prod_i \lambda_i$.

For symmetric matrices, **all eigenvalues are real** and there exists an orthonormal eigenbasis.

## The spectral theorem (symmetric case)

If $A = A^\top \in \mathbb{R}^{n \times n}$:

$$
A = Q \Lambda Q^\top = \sum_{i=1}^n \lambda_i q_i q_i^\top
$$

where $Q = [q_1 | \dots | q_n]$ is orthogonal ($Q^\top Q = I$) and $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$.

Geometric meaning: in the eigenbasis $\{q_i\}$, the action of $A$ is independent scaling along each axis. Symmetric matrices have **no rotational component**. They are pure stretches in some orthogonal frame.

<!-- visual:spectral-theorem-change-basis -->
<figure class="learning-figure" aria-labelledby="spectral-change-basis-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="spectral-change-basis-title">Read the spectral theorem as rotate into the eigenbasis, scale each coordinate, then rotate back.</p>
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
			<path class="viz-baseline" d="M106 148L234 66 M106 66L234 148"></path>
			<text class="viz-axis-label" x="238" y="66">q₁ = (1, 1)/√2</text>
			<text class="viz-axis-label" x="24" y="76">q₂ = (−1, 1)/√2</text>
			<path class="viz-roc-curve" d="M170 130L246 92" marker-end="url(#spectral-arrow-input)"></path>
			<text class="viz-callout" x="253" y="94">v</text>
			<text class="viz-label" x="24" y="153">z = Qᵀv = (3/√2, −1/√2)</text>
			<rect class="viz-plot-bg" x="12" y="178" width="336" height="176" rx="4"></rect>
			<text class="viz-callout" x="24" y="200">2 · Scale independently with Λ = diag(3, 1)</text>
			<path class="viz-axis" d="M58 316H326 M74 334V220"></path>
			<text class="viz-axis-label" x="306" y="332">q₁</text>
			<text class="viz-axis-label" x="48" y="228">q₂</text>
			<path class="viz-operating-guide" d="M74 316V330 M142 316V330 M278 316V330 M142 330H278"></path>
			<path class="viz-roc-curve" d="M74 316L142 330" marker-end="url(#spectral-arrow-input)"></path>
			<circle class="viz-node" cx="142" cy="330" r="4"></circle>
			<text class="viz-label" x="96" y="285">before: z</text>
			<path class="viz-pr-curve" d="M74 316L278 330" marker-end="url(#spectral-arrow-focus)"></path>
			<circle class="viz-operating-point" cx="278" cy="330" r="4"></circle>
			<text class="viz-callout" x="207" y="277">after: Λz</text>
			<text class="viz-label" x="24" y="349">q₁ coordinate ×3 · q₂ coordinate ×1</text>
			<rect class="viz-plot-bg" x="12" y="368" width="336" height="160" rx="4"></rect>
			<text class="viz-callout" x="24" y="390">3 · Return to the original axes with Q</text>
			<path class="viz-axis" d="M56 498H318 M94 516V408"></path>
			<path class="viz-baseline" d="M94 498L157 467"></path>
			<text class="viz-label" x="160" y="473">v = (2, 1)</text>
			<path class="viz-pr-curve" d="M94 498L264 418" marker-end="url(#spectral-arrow-focus)"></path>
			<text class="viz-callout" x="216" y="410">Av = (5, 4)</text>
			<text class="viz-label" x="24" y="521">Q(Λz) = QΛQᵀv = (5, 4)</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> <var>Q</var><sup>T</sup> does not stretch <var>v</var>; it only reports <var>v</var> along the orthogonal eigenvector axes. <var>Λ</var> then triples the q₁ coordinate and leaves the q₂ coordinate unchanged. <var>Q</var> returns those scaled coordinates to the original axes, giving <var>Av</var> = (5, 4).</figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-25-symmetric-matrices-and-positive-definiteness/">MIT OpenCourseWare 18.06</a> and Sheldon Axler's open-access <a href="https://linear.axler.net/">Linear Algebra Done Right</a>.</p>

## Connection to SVD

For symmetric positive semi-definite $A$: SVD and eigendecomposition coincide ($U = V = Q$, $\Sigma = \Lambda$). For general matrices they differ. SVD is the more general tool; eigendecomposition is the specialized one for symmetric / square matrices.

## Where eigenvalues show up in ML

| Object | What its eigenvalues tell you |
|--------|------------------------------|
| Covariance matrix | Variances along principal axes (PCA) |
| Hessian of loss | Local curvature; condition number = $\lambda_\max / \lambda_\min$ |
| Graph Laplacian | Connectivity, spectral clustering, GNN smoothness |
| Markov transition matrix | Mixing rate (second-largest eigenvalue) |
| Attention $Q K^\top$ | Effective rank; low-rank structure |
| Recurrent weight matrix | Whether RNN gradients explode/vanish |

## Common pitfalls

- **Treating asymmetric matrices like symmetric ones.** Asymmetric matrices may have complex eigenvalues and may not be diagonalizable at all (Jordan form).
- **Computing eigendecomposition for huge matrices.** Use Lanczos / Arnoldi or randomized SVD for large-scale; full eigendecomposition is $O(n^3)$.
- **Confusing eigenvalues with singular values.** Equal only for symmetric PSD matrices; otherwise singular values are $\sqrt{\lambda_i(A^\top A)}$.

## Related

- [SVD and PCA](/concepts/svd-and-pca/). Generalization to all matrices.
- [Positive definite matrices](/concepts/positive-definite-matrices/). The cone of PSD matrices.
