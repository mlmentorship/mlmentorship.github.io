---
title: "Monte Carlo and importance sampling"
description: "Estimate expectations by averaging over random samples. The simplest way to compute integrals you can't compute analytically."
date: "2025-10-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Monte Carlo** estimates $\mathbb{E}_p[f(X)] = \int f(x) p(x)\, dx$ by drawing $X_1, \dots, X_n \sim p$ and computing the sample average $\hat\mu = \tfrac{1}{n} \sum_i f(X_i)$. **Importance sampling** corrects for sampling from a different distribution $q$ by reweighting: $\hat\mu = \tfrac{1}{n} \sum_i f(X_i) \tfrac{p(X_i)}{q(X_i)}$.

Almost every probabilistic ML algorithm computes intractable expectations: posterior expectations in Bayesian inference, gradients of expectations in REINFORCE, off-policy returns in RL. Monte Carlo and importance sampling are the universal hammers when you can sample from the relevant distribution but can't integrate analytically.

## Plain Monte Carlo

If $X_i \sim p$ are i.i.d.:

- **Unbiased**: $\mathbb{E}[\hat\mu] = \mathbb{E}_p[f(X)]$.
- **Variance**: $\mathrm{Var}(\hat\mu) = \mathrm{Var}_p(f) / n$.
- **Convergence rate**: $\sqrt{n}$ regardless of dimension. (Beats deterministic numerical integration in high dimensions.)

The CLT gives Gaussian confidence intervals: $\hat\mu \pm 1.96 \cdot \hat\sigma / \sqrt{n}$.

## Importance sampling

When sampling from $p$ is hard (e.g., posterior, rare event), sample from a proposal $q$ instead:

$$
\mathbb{E}_p[f(X)] = \mathbb{E}_q\!\left[ f(X) \frac{p(X)}{q(X)} \right] \approx \frac{1}{n} \sum_i f(X_i) w_i, \quad w_i = \frac{p(X_i)}{q(X_i)}.
$$

The weights $w_i$ are **importance weights**. Critical:

- **$q$ must be positive wherever $p \cdot f$ is non-zero** (no holes).
- The variance of the estimator depends on $\mathrm{Var}_q[(f \cdot p/q)]$. A poorly matched $q$ can make variance enormous.
- **Self-normalized IS**: when $p$ is only known up to a constant, use $\hat\mu = \sum w_i f(X_i) / \sum w_i$. Biased (consistent) but always usable.

## Effective sample size

A diagnostic for how well IS is working:

$$
\mathrm{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}.
$$

If most weight concentrates on a single sample, ESS $\approx 1$ even when $n = 10^4$. Check ESS before trusting an IS estimate.

## Where it shows up in ML

| Use case | What's the integral |
|----------|---------------------|
| REINFORCE policy gradient | $\nabla_\theta \mathbb{E}_{\pi_\theta}[R]$ |
| Variational inference (ELBO gradient via reparam alternatives) | $\nabla_\phi \mathbb{E}_{q_\phi}[\log p - \log q]$ |
| Off-policy RL (importance sampling correction) | $\mathbb{E}_{\pi_\text{behavior}}[\tfrac{\pi_\text{target}}{\pi_\text{behavior}} R]$ |
| Bayesian posterior predictive | $\int p(y \mid \theta) p(\theta \mid D)\, d\theta$ |
| Importance-weighted autoencoders (IWAE) | tighter ELBO via $K$-sample IS |

## Variance reduction

- **Control variates**: subtract a quantity with known mean: $f(X) - c(g(X) - \mathbb{E}[g])$.
- **Antithetic variates**: pair $X$ with $-X$ (for symmetric $p$).
- **Stratified sampling**: divide the domain into strata and sample within each.
- **Rao–Blackwellization**: condition out variables you can integrate exactly.

## Common pitfalls

- **Heavy-tailed importance weights.** Variance can be infinite even when the true expectation exists.
- **Reusing samples across $n$ proposals** (e.g., multi-step IS in off-policy RL): the weights compound, and variance explodes.
- **Confusing self-normalized IS bias with MC bias.** Plain MC is unbiased; self-normalized IS is biased but consistent.
- **Forgetting that $q$ must dominate $p \cdot f$.** A "small" hole in $q$ where $p$ has mass introduces unbounded bias.
