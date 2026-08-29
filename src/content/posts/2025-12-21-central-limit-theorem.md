---
title: "Central limit theorem"
description: "Sums of many independent random variables become Gaussian. Why nearly every error bar in ML and statistics is computed from a normal distribution."
date: "2025-12-21"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

If $X_1, \dots, X_n$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2$, then as $n \to \infty$:

$$
\sqrt{n} \cdot (\bar X_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2).
$$

The standardized sample mean converges in distribution to a Gaussian, regardless of the original distribution's shape (as long as variance is finite).

The CLT is why we can:

- Build Gaussian-based confidence intervals for almost any estimator (sample mean, regression coefficients, A/B test deltas).
- Use $z$-tests and $t$-tests on data that isn't itself Gaussian.
- Trust that with enough samples, our reported metric ± std is approximately calibrated.

It also explains the prevalence of Gaussian assumptions in ML. Many quantities are sums or averages, and so naturally trend Gaussian.

## What "enough samples" means

The Berry–Esseen theorem bounds how fast convergence happens:

$$
\sup_x \big| F_n(x) - \Phi(x) \big| \le \frac{C \cdot \mathbb{E}[|X - \mu|^3]}{\sigma^3 \sqrt{n}}
$$

with $C \approx 0.4$. For symmetric, light-tailed distributions, $n = 30$ already gives an excellent Gaussian approximation. For heavy-tailed or skewed distributions you may need $n = 1000$ or more.

Heuristic check: plot a histogram of bootstrap means; if it looks Gaussian, the CLT has kicked in.

## Variants

- **Multivariate CLT**: $\sqrt{n}(\bar X_n - \mu) \to \mathcal{N}(0, \Sigma)$ for a vector-valued sum with covariance matrix $\Sigma$.
- **Lyapunov / Lindeberg CLT**: relaxes the i.i.d. assumption to independent (not identical) with mild moment conditions.
- **Martingale CLT**: extends to dependent data forming a martingale; used in online learning regret analysis.
- **CLT for U-statistics, M-estimators**: extends to functions of multiple samples.

## When CLT fails

- **Infinite variance** (e.g., Cauchy distribution): CLT does not apply; sample means do not concentrate. Use stable distributions instead.
- **Strong dependence**: highly correlated samples violate the i.i.d. assumption; effective sample size is much less than $n$.
- **Discrete distributions on small support**: CLT applies but the discrete approximation may be visibly bad until $n$ is large.

## Common pitfalls

- **Treating $n = 30$ as universally enough.** It is not for skewed or heavy-tailed data.
- **Using CLT on small sample sizes for inference.** Below $n \approx 30$, prefer the $t$-distribution (which uses the sample-estimated variance) rather than $z$.
- **Using parametric CLT confidence intervals on data that isn't independent.** A/B tests with user-level dependence (one user, multiple events) have effective $n$ much smaller than event count; cluster-bootstrap or use mixed-effects models.
- **Confusing "the mean is normally distributed" with "the data is normally distributed."** The CLT is about the *mean*, not individual samples.
