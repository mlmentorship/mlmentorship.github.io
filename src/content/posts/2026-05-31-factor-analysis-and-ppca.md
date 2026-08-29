---
title: "Factor analysis and probabilistic PCA"
description: "Factor analysis uses latent factors with per-feature noise. Probabilistic PCA uses isotropic noise and recovers classical PCA in its zero-noise limit."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Factor analysis (FA) is a **latent linear-Gaussian model**: each observation is a linear map of a few low-dimensional latent factors plus Gaussian noise. **Probabilistic PCA (PPCA)** is the special case with **isotropic** noise, and classical PCA falls out as its zero-noise / maximum-likelihood limit.

This is the model that turns PCA from "an eigen-decomposition trick" into "a probabilistic generative model," which is the framing senior interviewers want. It connects dimensionality reduction to the EM algorithm, to VAEs (a nonlinear PPCA), and to the generative-vs-discriminative discussion. It's also a clean example of how a **prior + likelihood** recovers a classical algorithm as a limiting case.

## The generative model

Latent factor $\mathbf{z} \in \mathbb{R}^k$ with $k \ll d$, observation $\mathbf{x} \in \mathbb{R}^d$:

$$
\mathbf{z} \sim \mathcal{N}(0, I), \qquad \mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(W\mathbf{z} + \boldsymbol{\mu},\ \Psi).
$$

$W \in \mathbb{R}^{d\times k}$ is the **factor loading matrix** (the directions), and $\Psi$ is the noise covariance. Marginalizing $\mathbf{z}$ gives a Gaussian with **low-rank-plus-structured** covariance:

$$
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu},\ WW^\top + \Psi).
$$

The whole model is the claim: *the correlations between observed variables are explained by a few shared latent factors; whatever is left is independent per-feature noise.*

## FA vs PPCA vs PCA: it's all about $\Psi$

| Model | Noise covariance $\Psi$ | Consequence |
| --- | --- | --- |
| **Factor analysis** | **diagonal** $\text{diag}(\psi_1,\dots,\psi_d)$ | per-feature noise; **scale-invariant**; models unique variances |
| **Probabilistic PCA** | **isotropic** $\sigma^2 I$ | one shared noise level; MLE has closed form via eigendecomposition |
| **Classical PCA** | $\sigma^2 \to 0$ limit | deterministic projection onto top-$k$ eigenvectors |

For interviews, distinguish the noise models: **FA has diagonal noise covariance, while PPCA uses the same isotropic noise for every feature.** FA is invariant to rescaling individual features. PCA and PPCA are sensitive to feature scaling, which is why inputs are usually standardized first.

## Fitting it

- **PPCA** has a **closed-form MLE**: $W$ is recovered from the top-$k$ eigenvectors of the sample covariance scaled by $(\lambda_i - \sigma^2)^{1/2}$, with $\sigma^2$ = average of the discarded eigenvalues. So PPCA ≈ PCA plus a noise estimate.
- **FA** has no closed form (the diagonal $\Psi$ couples things); it's fit with **EM**: the E-step infers the posterior over factors $p(\mathbf{z}\mid\mathbf{x})$, the M-step updates $W$ and $\Psi$. This is a textbook EM application.

## Why the probabilistic version is worth it

Recasting PCA as a model buys you things plain PCA can't do:

- A proper **likelihood** → principled model comparison and a way to choose $k$.
- Natural handling of **missing data** (marginalize unobserved dimensions in EM).
- A generative model you can **sample** from.
- **Mixtures of PPCA/FA** for non-linear, multi-modal structure.
- The conceptual bridge to the **VAE**, which is "PPCA with a neural-network decoder and amortized inference."

## What an interviewer expects you to say

1. Write the **latent linear-Gaussian generative model** and the marginal covariance $WW^\top + \Psi$.
2. State the difference: **FA = diagonal noise, PPCA = isotropic noise, PCA = zero-noise limit of PPCA**.
3. Explain the practical consequence: **FA is scale-invariant; PCA/PPCA require feature standardization**.
4. Know that **PPCA has a closed-form (eigendecomposition) MLE** while **FA needs EM**.
5. Bonus: connect to **VAEs** (nonlinear PPCA) and note the probabilistic framing enables missing data, model selection, and sampling.

## Common confusions

- **"FA and PCA are the same."** FA models per-feature (diagonal) noise and explains *covariance*; PCA maximizes *retained variance* and assumes isotropic/zero noise. They give different loadings unless noise is uniform.
- **"PPCA is fancier PCA with no payoff."** The payoff is the likelihood: model selection, missing data, sampling, mixtures.
- **"The factors are unique."** $W$ is only identifiable up to rotation (you can rotate $\mathbf{z}$ and absorb it into $W$), hence "factor rotation" (varimax) for interpretability.
- **"FA needs scaling like PCA."** FA is invariant to per-feature rescaling because its diagonal noise absorbs scale; PCA is not.

---

*Related: [SVD and PCA](/concepts/svd-and-pca/), [Expectation-maximization](/concepts/expectation-maximization/), [Gaussian mixture models](/concepts/gaussian-mixture-models/), [Variational autoencoders](/concepts/variational-autoencoders/).*
