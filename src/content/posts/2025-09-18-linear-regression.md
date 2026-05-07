---
title: "Linear regression"
description: "Predict a continuous target as a linear combination of features by minimizing squared error. Closed-form solution, MLE under Gaussian noise, and the foundation everything else builds on."
date: "2025-09-18"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Linear regression models $y = w^\top x + b + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. The MLE / least-squares estimator is

$$
\hat w = (X^\top X)^{-1} X^\top y.
$$

## Why it matters

Linear regression is the most-analyzed model in statistics and the building block for almost everything: GLMs, kernel ridge regression, MLP last layers, factor models. Knowing its assumptions and failure modes is essential. If you don't know when OLS is wrong, you don't know when fancier models help.

## Ordinary least squares (OLS)

Loss: $L(w, b) = \sum_i (y_i - w^\top x_i - b)^2 = \|y - Xw\|^2$ (absorb $b$ into $w$ by adding a column of 1s).

Closed-form minimizer (when $X^\top X$ is invertible):

$$
\hat w = (X^\top X)^{-1} X^\top y.
$$

This is the **normal equations** solution. Derived by setting $\nabla_w L = 0$.

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
