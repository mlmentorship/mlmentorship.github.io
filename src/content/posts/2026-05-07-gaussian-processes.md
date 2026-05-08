---
title: "Gaussian processes"
description: "A distribution over functions defined entirely by a covariance kernel. Predicts both a mean and a calibrated uncertainty. Beautiful theory, brutal scaling."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A **Gaussian process** (GP) places a prior over functions $f: \mathcal{X} \to \mathbb{R}$ such that any finite collection of values $\{f(x_1), \dots, f(x_n)\}$ is jointly Gaussian. The GP is fully specified by a mean function $m(x)$ (usually 0) and a covariance kernel $k(x, x')$.

## Why it matters

GPs are the canonical Bayesian regression model. They give a closed-form posterior over functions, with a predictive mean and a predictive variance per query point. The variance is calibrated and grows away from training data, which makes GPs the standard tool for **Bayesian optimization**, **active learning**, and any setting where uncertainty quantification matters as much as the prediction.

The cost is brutal scaling: $O(n^3)$ exact inference. GPs are practical up to a few thousand training points without approximation; modern variants (sparse, structured kernel, deep kernel) push that to millions.

## The mechanism

For training inputs $X$, training targets $\mathbf{y}$, test point $x_*$, prior covariance kernel $k$, and noise variance $\sigma^2$:

The joint distribution is

$$
\begin{bmatrix} \mathbf{y} \\ f(x_*) \end{bmatrix} \sim \mathcal{N}\!\left(\mathbf{0}, \begin{bmatrix} K + \sigma^2 I & k_* \\ k_*^\top & k_{**} \end{bmatrix}\right),
$$

where $K_{ij} = k(x_i, x_j)$, $k_*$ is the vector of $k(x_*, x_i)$, and $k_{**} = k(x_*, x_*)$.

Conditioning on training data gives the Gaussian posterior:

$$
\begin{aligned}
\mu(x_*) &= k_*^\top (K + \sigma^2 I)^{-1} \mathbf{y}, \\
\Sigma(x_*) &= k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*.
\end{aligned}
$$

Two matrix-vector products against $(K + \sigma^2 I)^{-1}$ give you both the predictive mean and the predictive variance at any test point. The hard part is $(K + \sigma^2 I)^{-1}$, which costs $O(n^3)$ to compute and $O(n^2)$ to store.

## Choosing the kernel

The kernel encodes prior beliefs about the function:

- **RBF / squared exponential**: $k(x, x') = \sigma_f^2 \exp(-\|x - x'\|^2 / 2 \ell^2)$. Smooth functions, infinite differentiability. Default.
- **Matern**: lets you tune smoothness via a half-integer parameter $\nu$. $\nu = 5/2$ is a common modern default; smoother than $\nu = 3/2$, less restrictive than RBF.
- **Periodic**: $k(x, x') = \sigma_f^2 \exp(-2 \sin^2(\pi |x - x'| / p) / \ell^2)$. For periodic data.
- **Linear**: $k(x, x') = x^\top x'$. Recovers Bayesian linear regression.
- **Sums and products** of valid kernels are valid kernels. Add a periodic and an RBF for "trend plus seasonality."

Hyperparameters (lengthscale $\ell$, variance $\sigma_f^2$, noise $\sigma^2$) are typically learned by maximizing the marginal likelihood:

$$
\log p(\mathbf{y} \mid X, \theta) = -\tfrac{1}{2} \mathbf{y}^\top (K + \sigma^2 I)^{-1} \mathbf{y} - \tfrac{1}{2} \log |K + \sigma^2 I| - \tfrac{n}{2} \log 2\pi.
$$

## Scaling: sparse and approximate variants

- **Inducing points** ([Snelson & Ghahramani, 2006](https://papers.nips.cc/paper_files/paper/2005/hash/4491777b1aa8b5b32c2e8666dbe1a495-Abstract.html)). Pick $m \ll n$ inducing inputs, approximate $K$ via a low-rank decomposition. $O(n m^2)$ training, $O(m^2)$ prediction.
- **SVGP** ([Hensman et al., 2013](https://arxiv.org/abs/1309.6835)). Variational inference over inducing point values. Stochastic mini-batch training. The standard for large-data GPs.
- **KISS-GP / structured kernels** ([Wilson & Nickisch, 2015](https://arxiv.org/abs/1503.01057)). Exploit grid structure for $O(n)$ inference.
- **Deep kernels**. Replace $k(x, x')$ with $k(\phi(x), \phi(x'))$ where $\phi$ is a neural network. Combines deep features with calibrated uncertainty.

## Where GPs are still the right tool

- **Bayesian optimization** of expensive functions (hyperparameter search, materials discovery, drug design). The acquisition function uses the posterior variance to balance exploration and exploitation.
- **Geospatial / time-series modeling** with structured covariance (kriging is just a GP).
- **Small-data regression** where calibrated uncertainty matters more than predictive accuracy.
- **Probabilistic numerics** (treat numerical algorithms as Bayesian inference).

For most large-scale supervised learning, deep nets with bootstrapping or deep ensembles deliver competitive uncertainty estimates at much better scaling.

## Common pitfalls

- **Using RBF without setting the lengthscale**. Default lengthscale produces nearly-flat predictions or wildly oscillating ones. Always optimize.
- **Ignoring the noise term $\sigma^2$**. Without it, $K$ is often singular and inversion fails. Add a small jitter even for noiseless data.
- **Reading posterior variance as "model uncertainty."** GP variance is uncertainty in the function value given the kernel and the prior. Misspecified kernels give miscalibrated variance.
- **Treating GPs as automatic.** Kernel choice is a strong prior. The model can fail silently if the kernel does not match the data structure.

## Related

- [Kernel methods and the kernel trick](/concepts/kernel-methods-and-the-kernel-trick/).
- [Bayes' rule and the posterior](/concepts/bayes-rule-and-the-posterior/).
- [Maximum likelihood estimation](/concepts/maximum-likelihood-estimation/).
