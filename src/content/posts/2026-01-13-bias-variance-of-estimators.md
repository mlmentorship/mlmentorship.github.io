---
title: "Bias and variance of estimators"
description: "An estimator has bias (systematic error) and variance (sample-to-sample wobble). Mean-squared error decomposes into the two."
date: "2026-01-13"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

For an estimator $\hat\theta$ of a parameter $\theta$, **bias** is $\mathbb{E}[\hat\theta] - \theta$ and **variance** is $\mathbb{E}[(\hat\theta - \mathbb{E}[\hat\theta])^2]$. Mean-squared error decomposes as $\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Variance}$.

This decomposition is the statistical version of the [bias–variance tradeoff](/questions/bias-variance-tradeoff/) familiar from ML: complex models have low bias but high variance; simple models have high bias but low variance. The same accounting applies to any estimator. Sample mean, regularized regression coefficient, importance sampling weight.

## The decomposition

For estimator $\hat\theta$ of a fixed (non-random) parameter $\theta$:

$$
\begin{aligned}
\mathrm{MSE}(\hat\theta) &= \mathbb{E}[(\hat\theta - \theta)^2] \\
&= \mathbb{E}[(\hat\theta - \mathbb{E}\hat\theta)^2] + (\mathbb{E}\hat\theta - \theta)^2 \\
&= \mathrm{Var}(\hat\theta) + \mathrm{Bias}(\hat\theta)^2.
\end{aligned}
$$

The cross-term vanishes because $\mathbb{E}[\hat\theta - \mathbb{E}\hat\theta] = 0$. Two-line derivation; central to all of statistics.

## Why biased estimators can be useful

Unbiased estimators ($\mathbb{E}\hat\theta = \theta$) are not always optimal. A biased estimator with much lower variance can have lower MSE.

Examples:

| Estimator | Bias | Variance | When better |
|----------|------|----------|-------------|
| Sample mean | 0 | $\sigma^2/n$ | universal |
| Sample variance with $n-1$ | 0 | larger | unbiased baseline |
| Sample variance with $n$ (MLE) | small negative | smaller | when minimizing MSE |
| Ridge regression | nonzero | smaller than OLS | when $X^\top X$ is ill-conditioned |
| Stein estimator | shrinkage bias | strictly lower | always for $\ge 3$ dimensions |

The James-Stein estimator (1961) famously dominates the sample mean in $\ge 3$ dimensions despite being biased.

## Connection to ML model selection

In supervised learning, the same decomposition holds for the prediction error of a model:

$$
\mathbb{E}[(y - \hat f(x))^2] = \mathrm{Var}(\hat f(x)) + \mathrm{Bias}(\hat f(x))^2 + \sigma_\varepsilon^2.
$$

The $\sigma_\varepsilon^2$ term is irreducible noise. Increasing model capacity decreases bias but increases variance; regularization shifts the tradeoff toward higher bias.

## Common pitfalls

- **Equating "biased" with "bad."** Many useful estimators are biased; lower MSE is what matters.
- **Reporting variance without specifying what's random.** "Variance of the estimator" is over re-sampling the data; "variance of the prediction" is over both data and inputs. Different objects.
- **Forgetting that the cross-term vanishes only against the *expectation* of $\hat\theta$.** Random fixed offsets ruin the decomposition.
- **Confusing estimator variance with model variance.** Estimator variance is a property of the estimation procedure; model variance is a property of the model class.
