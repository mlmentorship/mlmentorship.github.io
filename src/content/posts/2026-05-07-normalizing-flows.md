---
title: "Normalizing flows"
description: "Generative models built from invertible transformations. Compute exact likelihoods and sample efficiently. At the cost of architectural restrictions."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A **normalizing flow** transforms a simple base distribution (typically standard Gaussian) into a target distribution through a sequence of **invertible**, **differentiable** mappings $f_K \circ \dots \circ f_1$. The change-of-variables formula gives exact log-likelihood:

$$
\log p_X(x) = \log p_Z(f^{-1}(x)) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|.
$$

## Why it matters

Flows are the only family of deep generative models that simultaneously offer:

- **Exact likelihoods** (unlike VAEs and diffusion, which give bounds).
- **Efficient sampling** (single forward pass, unlike diffusion's iterative denoise).
- **Tractable posterior** (the inverse function gives the exact $z$ for any $x$).

The architectural restriction (each layer must be invertible with a tractable Jacobian) limits expressiveness. Flows have been displaced by diffusion for high-fidelity image generation but remain useful for likelihood-critical applications: density estimation, anomaly detection, simulation-based inference, molecular generation.

## The change-of-variables formula

For an invertible $f$ with $z = f^{-1}(x)$:

$$
p_X(x) = p_Z(z) \cdot \left| \det J_{f^{-1}}(x) \right| = p_Z(z) \cdot \left| \det J_f(z) \right|^{-1}.
$$

For a composition of $K$ flows, the log-determinant decomposes additively:

$$
\log p_X(x) = \log p_Z(z) - \sum_{k=1}^{K} \log \left| \det J_{f_k}(z_{k-1}) \right|.
$$

The engineering challenge: design each $f_k$ to be (a) invertible, (b) expressive, and (c) have a **cheap-to-compute log-determinant**.

## Common flow families

| Family | Idea | Tradeoff |
|--------|------|----------|
| **Affine coupling** (NICE, RealNVP, Glow) | Split $x$ in half; one half passes through, the other is affinely transformed by a function of the first | Triangular Jacobian → $\det$ is product of diagonal; needs many layers for expressiveness |
| **Autoregressive** (MAF, IAF) | Each output dimension is an affine function of preceding ones; Jacobian is triangular | MAF: fast density, slow sample. IAF: fast sample, slow density. |
| **Continuous-time / Neural ODE** (FFJORD) | Define $f$ via $dx/dt = g_\theta(x, t)$ and integrate; Jacobian via Hutchinson trace estimator | Very expressive; expensive integration |
| **Invertible 1×1 convolutions** (Glow) | Permutation generalization for image flows | Used inside Glow for permutation between coupling layers |

## RealNVP / coupling layers (the workhorse)

Split $x = (x_a, x_b)$. Then:

$$
y_a = x_a, \qquad y_b = x_b \odot \exp(s(x_a)) + t(x_a)
$$

with neural nets $s, t$. The Jacobian is lower triangular with $\exp(s(x_a))$ on the diagonal of the $y_b$ block. Determinant: $\prod_i \exp(s(x_a)_i)$.

Stack many coupling layers, alternating which half passes through, with shuffles or 1×1 convs between them.

## When to use flows in 2026

| Setting | Flows vs. alternatives |
|---------|----------------------|
| High-fidelity image generation | Use diffusion; flows are non-competitive |
| Density estimation, OOD detection | Flows give exact likelihood |
| Simulation-based inference (likelihood-free) | Flows excellent (NPE, NRE) |
| Molecular conformation / coordinates | Flows used (E-NF, equivariant flows) |
| Probabilistic forecasting | Flows + RNN backbones (Real-NVP-style) |
| Variational inference posterior approx | Flows as flexible $q$ |

## Common pitfalls

- **Computing log-determinants without exploiting structure.** General log-det is $O(d^3)$. Always use a flow with a structured Jacobian (triangular, low-rank).
- **Confusing log $|\det|$ with log-likelihood directly.** The full formula has both the base density term and the determinant term.
- **Treating flows as fast-to-train.** They are usually slower per epoch than VAE / diffusion at matched parameter count due to expensive Jacobian computations.
- **Using flows on discrete data.** Flows assume continuous, differentiable spaces. For discrete: dequantize (add uniform noise), or use discrete normalizing flows (more complex).

## Related

- [Variational autoencoders](/reference/variational-autoencoders/). Alternative latent-variable generative model.
- [Autoregressive vs. diffusion](/reference/autoregressive-vs-diffusion/). Broader paradigm comparison.
