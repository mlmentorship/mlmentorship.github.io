---
title: "SVM and the kernel trick"
description: "Maximum-margin classifier with a kernel that lets it operate in implicit high-dimensional feature spaces. Beautiful theory; less common in 2026 production."
date: "2025-11-16"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A **Support Vector Machine** finds the hyperplane $w^\top x + b = 0$ that maximally separates the two classes (largest margin). The **kernel trick** replaces $x$ with an implicit nonlinear feature map $\phi(x)$ that is never computed. Only the inner products $K(x, x') = \phi(x)^\top \phi(x')$ matter.

## Why it matters

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

- [Calibration](/reference/calibration/). SVM scores need Platt scaling for probabilities.
- [Linear regression](/reference/linear-regression/). Kernel ridge regression is the regression analog.
