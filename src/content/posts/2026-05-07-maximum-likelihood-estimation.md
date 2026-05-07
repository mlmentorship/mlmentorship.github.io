---
title: "Maximum likelihood estimation"
description: "The dominant statistical principle: pick parameters that make the observed data most probable. Reduces to minimizing cross-entropy for classification and MSE for Gaussian regression."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

For a parametric family $p(x \mid \theta)$ and observed data $\{x_1, \dots, x_n\}$, the **maximum likelihood estimate (MLE)** is

$$
\hat\theta_\text{MLE} = \arg\max_\theta \prod_{i=1}^{n} p(x_i \mid \theta) = \arg\max_\theta \sum_{i=1}^{n} \log p(x_i \mid \theta).
$$

## Why it matters

MLE underlies almost every modern ML loss function:

- Cross-entropy for classification = MLE under a categorical model.
- Mean-squared error = MLE under a Gaussian noise model.
- Negative log-likelihood for language models = MLE.

When you read "minimize the negative log-likelihood," you're reading MLE.

## Properties

Under regularity conditions (smooth log-likelihood, identifiable model, true parameter in interior), MLE is:

- **Consistent**: $\hat\theta \to \theta^*$ as $n \to \infty$.
- **Asymptotically normal**: $\sqrt{n}(\hat\theta - \theta^*) \to \mathcal{N}(0, I(\theta^*)^{-1})$ where $I(\theta)$ is the **Fisher information**.
- **Asymptotically efficient**: achieves the Cramér–Rao lower bound. No consistent estimator has lower asymptotic variance.

## Common cases

| Model | MLE solution |
|-------|-------------|
| Gaussian mean (known $\sigma^2$) | $\hat\mu = \bar x$ |
| Gaussian variance | $\hat\sigma^2 = \frac{1}{n} \sum (x_i - \bar x)^2$ (biased; sample-variance uses $n-1$) |
| Bernoulli ($p$) | $\hat p = \text{fraction of successes}$ |
| Categorical | empirical class frequencies |
| Linear regression with Gaussian noise | OLS: $\hat\beta = (X^\top X)^{-1} X^\top y$ |
| Logistic regression | no closed form; iterative (Newton, gradient methods) |

## Connection to cross-entropy and KL

For categorical $p_\theta(x)$ and empirical distribution $\hat p_\text{data}$, the negative log-likelihood divided by $n$ is

$$
-\tfrac{1}{n} \sum_i \log p_\theta(x_i) = H(\hat p_\text{data}, p_\theta) = H(\hat p_\text{data}) + \mathrm{KL}(\hat p_\text{data} \| p_\theta).
$$

Maximizing likelihood = minimizing cross-entropy = minimizing KL from the empirical distribution to the model.

## MLE vs MAP

MAP (maximum a posteriori) adds a prior: $\hat\theta_\text{MAP} = \arg\max_\theta \log p(\theta) + \sum \log p(x_i \mid \theta)$. MAP equals MLE when the prior is uniform (improper). Common choices:

- Gaussian prior $\Rightarrow$ L2 regularization on $\theta$.
- Laplace prior $\Rightarrow$ L1 regularization.

## Common pitfalls

- **Treating sample variance as MLE.** MLE for Gaussian variance divides by $n$ (biased); the sample variance divides by $n-1$ (unbiased). Different estimators.
- **Stopping at MLE without checking identifiability.** If two parameter values yield identical likelihoods, MLE is non-unique.
- **Trusting MLE on small samples.** Asymptotic guarantees can be misleading when $n$ is small relative to dimension; use cross-validation or Bayesian methods.
- **Forgetting that MLE on a misspecified model is still well-defined.** It converges to the parameter that minimizes KL to the (mismatched) true distribution within the model family.
