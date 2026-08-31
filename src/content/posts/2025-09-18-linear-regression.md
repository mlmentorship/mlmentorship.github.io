---
title: "Linear regression"
description: "Predict a continuous target as a linear combination of features by minimizing squared error. Closed-form solution, MLE under Gaussian noise, and the foundation everything else builds on."
date: "2025-09-18"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Linear regression models $y = w^\top x + b + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. The MLE / least-squares estimator is

$$
\hat w = (X^\top X)^{-1} X^\top y.
$$

Linear regression is the most-analyzed model in statistics and the building block for almost everything: GLMs, kernel ridge regression, MLP last layers, factor models. Knowing its assumptions and failure modes is essential. If you don't know when OLS is wrong, you don't know when fancier models help.

## Ordinary least squares (OLS)

Loss: $L(w, b) = \sum_i (y_i - w^\top x_i - b)^2 = \|y - Xw\|^2$ (absorb $b$ into $w$ by adding a column of 1s).

Closed-form minimizer (when $X^\top X$ is invertible):

$$
\hat w = (X^\top X)^{-1} X^\top y.
$$

This is the **normal equations** solution. Derived by setting $\nabla_w L = 0$.

<!-- visual:ols-orthogonal-projection -->
<figure class="learning-figure" aria-labelledby="ols-projection-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ols-projection-title">See why the least-squares residual must be perpendicular to the feature space.</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 285" role="img" aria-labelledby="ols-projection-svg-title ols-projection-svg-desc">
			<title id="ols-projection-svg-title">Ordinary least squares as an orthogonal projection</title>
			<desc id="ols-projection-svg-desc">A slanted plane represents the column space of the design matrix X. From the origin, the fitted vector y-hat ends on that plane. The observed target vector y ends above it. A dashed residual vector connects y-hat to y and meets the plane at a marked right angle. This perpendicular residual satisfies X transpose r equals zero.</desc>
			<defs>
				<marker id="ols-projection-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
					<path d="M0 0L7 3.5L0 7Z" style="fill:var(--viz-edge)"></path>
				</marker>
				<marker id="ols-residual-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
					<path d="M0 0L7 3.5L0 7Z" style="fill:var(--viz-focus-stroke)"></path>
				</marker>
			</defs>
			<path d="M24 206L224 105L337 155L137 256Z" style="fill:var(--viz-state-bg);fill-opacity:.72;stroke:var(--viz-state-stroke);stroke-width:2"></path>
			<text class="viz-axis-label" x="276" y="183">Col(X)</text>
			<text class="viz-label" x="277" y="199">all possible fitted vectors Xw</text>
			<circle cx="67" cy="222" r="4" style="fill:var(--c-text);stroke:var(--c-text)"></circle>
			<text class="viz-label" x="45" y="241">origin</text>
			<path d="M67 222L207 148" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;marker-end:url(#ols-projection-arrow)"></path>
			<text class="viz-axis-label" x="111" y="169">fitted y_hat = Xw_hat</text>
			<path d="M67 222L251 55" style="fill:none;stroke:var(--c-text-soft);stroke-width:2.4;marker-end:url(#ols-projection-arrow)"></path>
			<text class="viz-axis-label" x="154" y="102">observed y</text>
			<path d="M207 148L251 55" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-dasharray:6 4;marker-end:url(#ols-residual-arrow)"></path>
			<text class="viz-callout" x="249" y="113">residual r</text>
			<text class="viz-label" x="249" y="127">y - y_hat</text>
			<path d="M207 148L219 154L225 142" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
			<circle cx="207" cy="148" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle>
			<circle cx="251" cy="55" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--c-text-soft);stroke-width:2"></circle>
			<rect x="194" y="223" width="142" height="42" rx="5" style="fill:var(--viz-neutral-bg);stroke:var(--c-rule);stroke-width:1.5"></rect>
			<text class="viz-axis-label" x="265" y="241" text-anchor="middle">r is perpendicular to Col(X)</text>
			<text class="viz-callout" x="265" y="257" text-anchor="middle">X^T r = 0</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> every coefficient vector produces a fitted vector inside Col(X). OLS chooses the point y_hat closest to the observed y, so the leftover residual meets that space at a right angle. Writing that orthogonality as X<sup>T</sup>(y - Xw_hat) = 0 gives the normal equations. This is the geometry of the solution, not advice to form an inverse numerically: use QR or SVD as described below. Original schematic checked against <a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/positive-definite-matrices-and-applications/orthogonal-projections-and-their-applications/">MIT OpenCourseWare's projection notes</a> and the <a href="https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd141.htm">NIST least-squares handbook</a>.</figcaption>
</figure>

Numerically, never compute it that way. Use:

- **QR decomposition**: $X = QR$ then $\hat w = R^{-1} Q^\top y$. Stable.
- **SVD**: works even when $X^\top X$ is singular (gives minimum-norm solution).
- **Gradient descent**: for huge $n$ where matrix factorization doesn't fit.

## Probabilistic interpretation

OLS is MLE for $y \mid x \sim \mathcal{N}(w^\top x, \sigma^2)$. The Gaussian-noise assumption is what motivates the squared-error loss; if errors are heavy-tailed, OLS is no longer optimal (consider Huber loss or quantile regression).

## Ridge and lasso

When $X^\top X$ is ill-conditioned (collinear features, $p > n$), OLS variance explodes. Add regularization:

| Method | Penalty | Effect |
|--------|---------|--------|
| **Ridge** (L2) | $\lambda \|w\|_2^2$ | Shrinks all coefficients toward 0; closed form: $\hat w = (X^\top X + \lambda I)^{-1} X^\top y$ |
| **Lasso** (L1) | $\lambda \|w\|_1$ | Drives some coefficients to exactly 0 (sparsity); no closed form (use coordinate descent / proximal gradient) |
| **Elastic net** | both | Combines sparsity with grouping |

Ridge is the default for prediction; lasso for variable selection or interpretability.

## Assumptions and diagnostics

The classical OLS assumptions:

1. **Linearity**: $\mathbb{E}[y \mid x]$ is linear in $x$.
2. **Independence**: residuals are independent.
3. **Homoskedasticity**: residual variance is constant.
4. **Normality**: residuals are normally distributed (only matters for inference, not for prediction).

Check by plotting residuals: vs. predicted (linearity, homoskedasticity), vs. each feature (linearity), QQ plot (normality), Durbin-Watson (independence in time series).

## Gauss–Markov theorem

Under the first three assumptions (with finite variance), OLS is the **B**est **L**inear **U**nbiased **E**stimator (BLUE). Minimum variance among all linear unbiased estimators. Note: biased estimators (ridge) can do better in MSE.

## When to use vs. alternatives

- **Linear is enough**: low-dimensional clean data, interpretability needed.
- **Non-linear**: add basis functions (polynomial features), kernels (kernel ridge regression), or use trees / neural nets.
- **Heavy-tailed errors**: Huber regression, quantile regression.
- **Many irrelevant features**: lasso or elastic net.
- **Hierarchical / clustered data**: mixed-effects (linear mixed model).

## Common pitfalls

- **Inverting $X^\top X$ directly.** Use QR / SVD; the inverse is numerically unstable and ridge-style regularization is often needed.
- **Including highly collinear features without regularization.** Coefficients become unstable and uninterpretable; drop one or use ridge.
- **Reporting $R^2$ on training data and calling it generalization.** Use cross-validated $R^2$.
- **Confusing prediction intervals with confidence intervals.** Confidence intervals cover the *mean*; prediction intervals cover individual outcomes (much wider).
