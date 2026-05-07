---
title: "KL divergence"
description: "Asymmetric distance between probability distributions. Cross-entropy minus entropy. The mathematical glue holding most of probabilistic ML together."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

For probability distributions $p$ and $q$ over the same space:

$$
\mathrm{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \quad (\text{or } \int p \log(p/q)\, dx \text{ for continuous}).
$$

It's the expected log-ratio of $p$ to $q$ under $p$. Measuring how much information is lost using $q$ to encode samples from $p$.

## Why it matters

KL divergence is the fundamental object of statistical learning. It connects:

- Maximum likelihood (minimizing $\mathrm{KL}(\hat p_\text{data} \| p_\theta)$).
- Variational inference (minimizing $\mathrm{KL}(q_\phi \| p)$).
- Cross-entropy loss = entropy of data + KL.
- Information bottleneck and mutual information.
- Policy gradient methods in RL (TRPO, PPO use KL constraints).
- Knowledge distillation (student matches teacher distribution via KL).

## Properties

- **Non-negative**: $\mathrm{KL}(p \| q) \ge 0$, with equality iff $p = q$ (Gibbs' inequality).
- **Asymmetric**: $\mathrm{KL}(p \| q) \ne \mathrm{KL}(q \| p)$ in general. Choose direction based on whether you are "fitting $q$ to $p$" or vice versa.
- **Not a metric**: no triangle inequality, not symmetric.
- **Infinite if $q(x) = 0$ where $p(x) > 0$**: $q$ must cover the support of $p$.
- **Information-theoretic**: equals expected extra bits (or nats) per sample needed to encode $p$ using a code optimized for $q$.

## Forward vs. reverse KL

The asymmetry matters in practice. For approximating $p$ with $q$:

- **Forward KL**, $\mathrm{KL}(p \| q)$: penalizes $q$ for missing modes of $p$ ("**mean-seeking**". $q$ tries to cover all of $p$). Used in standard MLE.
- **Reverse KL**, $\mathrm{KL}(q \| p)$: penalizes $q$ for placing mass where $p$ has none ("**mode-seeking**". $q$ collapses to one mode). Used in variational inference.

For a multimodal $p$, forward KL gives a broad average; reverse KL picks one mode. Visualize on a 2-Gaussian mixture: forward = average ellipse over both; reverse = one of the two.

## Connection to cross-entropy

For empirical distribution $\hat p$ over a finite dataset:

$$
H(\hat p, p_\theta) = -\mathbb{E}_{\hat p}[\log p_\theta] = H(\hat p) + \mathrm{KL}(\hat p \,\|\, p_\theta).
$$

Cross-entropy = entropy + KL. Since entropy of the data doesn't depend on $\theta$, minimizing cross-entropy = minimizing KL = MLE. This is why "the loss is cross-entropy" and "we're minimizing KL to the data" are the same statement.

## Common usage in ML

| Use case | Direction |
|---------|-----------|
| Classification cross-entropy loss | Forward $\mathrm{KL}(\text{data} \| \text{model})$ |
| Variational inference (ELBO) | Reverse $\mathrm{KL}(q \| p)$ |
| RLHF / PPO penalty | Reverse $\mathrm{KL}(\pi_\theta \| \pi_\text{ref})$. Keep new policy close to reference |
| Knowledge distillation | Forward $\mathrm{KL}(\text{teacher} \| \text{student})$ with temperature |
| t-SNE | Reverse $\mathrm{KL}(P \| Q)$ on pairwise similarities |

## Common pitfalls

- **Computing KL between distributions with different supports.** If $p(x) > 0$ and $q(x) = 0$, KL is $+\infty$.
- **Confusing JS divergence (symmetric) with KL.** GANs originally used JS; modern variants (Wasserstein) avoid both.
- **Forgetting the asymmetry direction.** Forward and reverse KL produce qualitatively different optimizers.
- **Using KL on samples without density estimates.** KL is between distributions, not between sample sets; sample-based estimators are noisy and biased.
