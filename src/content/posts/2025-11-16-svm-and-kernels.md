---
title: "SVM and the kernel trick"
description: "Maximum-margin classifier with a kernel that lets it operate in implicit high-dimensional feature spaces. Beautiful theory; less common in 2026 production."
date: "2025-11-16"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **Support Vector Machine** finds the hyperplane $w^\top x + b = 0$ that maximally separates the two classes (largest margin). The **kernel trick** replaces $x$ with an implicit nonlinear feature map $\phi(x)$ that is never computed. Only the inner products $K(x, x') = \phi(x)^\top \phi(x')$ matter.

SVMs were the dominant classification method from ~1998 to ~2012, before deep learning took over for unstructured data and GBDT for tabular. They remain useful in low-data, high-dimensional regimes (small biology and physics datasets) and as a teaching example of margin maximization, convex optimization, and kernel methods.

The **kernel trick** itself remains relevant in Gaussian processes, kernel ridge regression, and modern theory (NTK).

## The hard-margin SVM (separable case)

Find $w, b$ minimizing $\|w\|^2$ subject to $y_i (w^\top x_i + b) \ge 1$ for all $i$, with $y_i \in \{-1, +1\}$. The constraint defines a margin of width $2 / \|w\|$; minimizing $\|w\|$ maximizes margin.

Convex quadratic program with linear constraints. Has a unique solution (for separable data).

## The soft-margin SVM (non-separable)

Allow some violations with slack variables $\xi_i \ge 0$:

$$
\min_{w, b, \xi}\; \tfrac{1}{2}\|w\|^2 + C \sum_i \xi_i \quad \text{s.t. } y_i(w^\top x_i + b) \ge 1 - \xi_i.
$$

$C$ trades margin width against violations. Equivalently, minimize the **hinge loss** $\max(0, 1 - y_i (w^\top x_i + b))$ plus L2 regularization.

## The dual formulation and the kernel trick

The dual problem is

$$
\max_\alpha \sum_i \alpha_i - \tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \quad \text{s.t. } 0 \le \alpha_i \le C, \sum_i \alpha_i y_i = 0.
$$

The data appears only as inner products $x_i^\top x_j$. Replace with $K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$ for any positive-definite kernel. Fits in the implicit feature space $\phi$ without ever computing it.

Common kernels:

| Kernel | $K(x, x')$ | Implicit feature space |
|--------|-----------|----------------------|
| Linear | $x^\top x'$ | original |
| Polynomial | $(x^\top x' + c)^d$ | all monomials of degree $\le d$ |
| RBF (Gaussian) | $\exp(-\gamma \|x - x'\|^2)$ | infinite-dimensional |
| Sigmoid | $\tanh(\alpha x^\top x' + c)$ | (not always PSD) |

## Support vectors

After training, the optimal $w = \sum_i \alpha_i y_i x_i$ (in primal) or its kernelized analog. Most $\alpha_i$ are zero; the points with $\alpha_i > 0$ are the **support vectors**. They sit on or inside the margin and entirely determine the decision boundary. Removing all non-support vectors leaves the model unchanged.

<!-- visual:svm-support-vectors-set-boundary -->
<figure class="learning-figure plot-panel" aria-labelledby="svm-support-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="svm-support-title">Which training points survive in the SVM decision function?</p>
	<svg viewBox="0 0 360 340" role="img" aria-labelledby="svm-support-svg-title svm-support-svg-desc">
		<title id="svm-support-svg-title">Support vectors on or inside two SVM margin rails</title>
		<desc id="svm-support-svg-desc">A vertical decision boundary lies between dashed negative-one and positive-one margin rails. Negative-class circles appear on the left and positive-class diamonds on the right. Three points on or inside the rails have prominent outer rings and labels saying alpha i is greater than zero, so they are support vectors. Four correctly classified points beyond the rails have no rings and labels saying alpha i equals zero. A note states that prediction sums only over the three ringed support vectors.</desc>
		<rect class="viz-plot-bg" x="12" y="30" width="336" height="240" rx="4"></rect>
		<text class="viz-axis-label" x="54" y="51">CLASS −1 · CIRCLES</text>
		<text class="viz-axis-label" x="306" y="51" text-anchor="end">CLASS +1 · DIAMONDS</text>
		<path class="viz-baseline" d="M120 58V246M240 58V246"></path>
		<path class="viz-axis" d="M180 58V246"></path>
		<text class="viz-label" x="116" y="66" text-anchor="end">f(x) = −1</text>
		<text class="viz-label" x="184" y="66">f(x) = 0</text>
		<text class="viz-label" x="244" y="66">f(x) = +1</text>
		<circle cx="58" cy="112" r="6" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle>
		<circle cx="74" cy="211" r="6" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle>
		<text class="viz-label" x="38" y="132">beyond margin</text>
		<text class="viz-callout" x="38" y="147">αᵢ = 0</text>
		<circle cx="120" cy="137" r="11" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></circle>
		<circle cx="120" cy="137" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle>
		<text class="viz-callout" x="108" y="160" text-anchor="end">on rail · αᵢ &gt; 0</text>
		<path d="M240 112L247 119L240 126L233 119Z" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
		<path d="M240 114L245 119L240 124L235 119Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></path>
		<text class="viz-callout" x="252" y="142">on rail · αᵢ &gt; 0</text>
		<circle cx="207" cy="198" r="11" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></circle>
		<path d="M207 192L213 198L207 204L201 198Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></path>
		<text class="viz-callout" x="219" y="220">inside · αᵢ &gt; 0</text>
		<path d="M302 94L308 100L302 106L296 100Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></path>
		<path d="M292 190L298 196L292 202L286 196Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></path>
		<text class="viz-label" x="322" y="166" text-anchor="end">beyond margin</text>
		<text class="viz-callout" x="322" y="181" text-anchor="end">αᵢ = 0</text>
		<rect class="viz-node viz-node--focus" x="30" y="282" width="300" height="40" rx="4"></rect>
		<text class="viz-callout" x="180" y="299" text-anchor="middle">decision(x) = sign(Σᵢ∈SV αᵢ yᵢ K(xᵢ, x) + b)</text>
		<text class="viz-node-value" x="180" y="315">prediction sums over the 3 ringed points only</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> first locate each point relative to its dashed class-margin rail. Correct points beyond the rail have zero hinge loss and <code>αᵢ = 0</code>, so they disappear from the fitted decision function. The three ringed points touch or enter the margin, have <code>αᵢ &gt; 0</code>, and alone support the boundary. For a separable hard-margin SVM they sit on the rails; soft margin also permits support vectors inside. Original schematic checked against <a href="https://link.springer.com/article/10.1023/A:1009715923555">Burges's SVM tutorial</a> and the <a href="https://scikit-learn.org/stable/modules/svm.html">scikit-learn SVM formulation</a>.</figcaption>
</figure>

## When to use SVMs in 2026

| Setting | SVM vs. alternative |
|---------|--------------------|
| Small high-dim data, clean features | RBF SVM still strong baseline |
| Tabular with categorical features | GBDT wins |
| Text / images / structured | Neural nets win |
| Huge data ($n > 10^5$) | SVMs scale poorly: $O(n^2)$ to $O(n^3)$ training |
| Online learning | Logistic / linear models |

For most 2026 production work, SVMs have been displaced. Their main uses are pedagogical and in legacy codebases.

## Common pitfalls

- **Forgetting to scale features.** RBF kernels are extremely sensitive to feature scale.
- **Tuning $C$ and $\gamma$ separately.** They interact; do a 2D grid search.
- **Calling SVM "non-parametric."** With the kernel trick the parameter count grows with the number of support vectors (effectively $O(n)$); behaves more like nearest-neighbor than like a fixed-parameter model.
- **Confusing hinge loss with logistic loss.** Hinge gives margins; logistic gives probabilities. SVM is not directly probabilistic without Platt scaling.

## Related

- [Calibration](/concepts/calibration/). SVM scores need Platt scaling for probabilities.
- [Linear regression](/concepts/linear-regression/). Kernel ridge regression is the regression analog.
