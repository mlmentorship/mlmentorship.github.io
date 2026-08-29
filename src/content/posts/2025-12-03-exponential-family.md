---
title: "Exponential family"
description: "A unified family of distributions (Gaussian, Bernoulli, Poisson, Beta, Gamma, etc.) with shared properties: sufficient statistics, conjugate priors, simple MLE."
date: "2025-12-03"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A distribution is in the **exponential family** if its density / mass function can be written as:

$$
p(x \mid \theta) = h(x) \exp\big( \eta(\theta)^\top T(x) - A(\theta) \big)
$$

with **natural parameter** $\eta(\theta)$, **sufficient statistic** $T(x)$, **base measure** $h(x)$, and **log-partition** $A(\theta)$ (which normalizes).

Most distributions you use day-to-day are exponential family: Gaussian, Bernoulli, categorical, Poisson, Beta, Gamma, Dirichlet, geometric, exponential. Recognizing them as such gives you free results:

- **MLE is closed-form** when the natural parameter is unconstrained: just match sample moments to model moments.
- **Conjugate priors** exist and are themselves exponential family.
- **Sufficient statistics** $T(x)$ contain all the data's information about $\theta$. You can summarize a dataset by $\sum_i T(x_i)$ and forget the rest.
- **Generalized linear models (GLMs)** are linear regression generalized to exponential-family responses.

## The canonical form

Given the form above:

- $T(x)$ are the **sufficient statistics** (Bernoulli: $T(x) = x$; Gaussian: $T(x) = (x, x^2)$).
- $\eta(\theta)$ are the **natural parameters** (Bernoulli: $\eta = \log\frac{p}{1-p}$, the logit; Gaussian: $\eta = (\mu/\sigma^2, -1/(2\sigma^2))$).
- $A(\theta)$ is the **log-partition function**; gradient gives the mean, Hessian gives the covariance of $T(x)$:

$$
\nabla A(\theta) = \mathbb{E}[T(X)], \qquad \nabla^2 A(\theta) = \mathrm{Cov}(T(X)).
$$

This is why MLE via moment-matching works: the gradient of the log-likelihood is "data sufficient stat minus model expected sufficient stat."

## Common members

| Distribution | Sufficient stat $T(x)$ | Natural parameter $\eta$ |
|--------------|------------------------|--------------------------|
| Bernoulli($p$) | $x$ | $\log(p/(1-p))$ (logit) |
| Categorical($\pi$) | one-hot$(x)$ | $\log \pi$ (log-probabilities) |
| Gaussian($\mu, \sigma^2$) | $(x, x^2)$ | $(\mu/\sigma^2, -1/(2\sigma^2))$ |
| Poisson($\lambda$) | $x$ | $\log \lambda$ |
| Beta($\alpha, \beta$) | $(\log x, \log(1-x))$ | $(\alpha - 1, \beta - 1)$ |
| Gamma($\alpha, \beta$) | $(\log x, x)$ | $(\alpha - 1, -\beta)$ |

## Generalized linear models

A **GLM** combines a linear predictor $\eta = X \beta$ with an exponential-family response distribution. The link function maps the linear predictor to the natural parameter:

| Response | GLM | Link |
|----------|-----|------|
| Continuous (Gaussian) | linear regression | identity |
| Binary (Bernoulli) | logistic regression | logit |
| Count (Poisson) | Poisson regression | log |
| Categorical | multinomial logistic | softmax |
| Time-to-event (Exponential, Weibull) | survival models | log |

Logistic regression *is* a GLM with Bernoulli response and logit link. This unifies the entire family of "regression-style" classifiers.

## Properties to remember

- **Convexity**: $A(\eta)$ is convex in $\eta$. So negative log-likelihood is convex, and there is a unique MLE.
- **Sufficient statistics**: by Pitman-Koopman-Darmois theorem, exponential families are essentially the only distributions with finite-dimensional sufficient statistics independent of $n$.
- **Conjugacy**: exponential families have conjugate priors, also in the exponential family.
- **Maximum entropy**: the exponential family with sufficient statistics $T$ matching given moments is the maximum-entropy distribution under those constraints.

## Common pitfalls

- **Forgetting that the Cauchy distribution is not exponential family.** Heavy tails break the sufficient-statistic property.
- **Confusing "exponential" (the distribution) with "exponential family" (the class).** The Exp($\lambda$) distribution is one member.
- **Treating natural parameters as the same as canonical parameters.** A Gaussian's natural parameters are $(\mu/\sigma^2, -1/(2\sigma^2))$, not $(\mu, \sigma^2)$. Some software libraries default to one or the other; check.
