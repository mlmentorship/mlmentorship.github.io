---
title: "Variational autoencoders (VAE)"
description: "Encode inputs to a latent distribution, decode samples back, optimize evidence lower bound. The cleanest gateway to deep generative models."
date: "2025-11-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A **variational autoencoder** [(Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114) is a generative model with a latent variable $z$, learned encoder $q_\phi(z \mid x)$, and decoder $p_\theta(x \mid z)$, trained to maximize the **evidence lower bound** (ELBO):

$$
\log p_\theta(x) \ge \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)] - \mathrm{KL}(q_\phi(z \mid x) \,\|\, p(z)).
$$

## Why it matters

VAEs introduced **amortized variational inference** to deep learning: a neural network learns to predict the posterior of a latent given the input, enabling end-to-end training of latent variable models with backprop. This idea now powers:

- Diffusion models (often built on top of a VAE in latent space. Stable Diffusion).
- Disentanglement research (β-VAE, factor-VAE).
- Generative pretraining for tabular and molecular data.
- Probabilistic recsys and time series.

The VAE is also the canonical example of the **reparameterization trick**, a tool used everywhere in modern probabilistic deep learning.

## The model

- **Prior**: $p(z) = \mathcal{N}(0, I)$.
- **Encoder**: $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma_\phi(x)^2))$. A neural net outputs mean and diagonal covariance.
- **Decoder**: $p_\theta(x \mid z)$. A neural net mapping $z$ back to a distribution over $x$ (Gaussian for continuous, Bernoulli for binary, categorical for discrete).

## The ELBO

The ELBO has two terms:

$$
\mathcal{L}_\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)]}_\text{reconstruction} - \underbrace{\mathrm{KL}(q_\phi(z | x) \,\|\, p(z))}_\text{regularizer}.
$$

- **Reconstruction**: rewards the decoder for assigning high probability to $x$ given samples from the encoder.
- **KL**: penalizes the encoder for diverging from the prior. Keeps the latent space compact and continuous.

Maximizing ELBO ≤ maximizing log-likelihood (true objective). The gap is $\mathrm{KL}(q_\phi \| p_\text{true posterior})$.

## The reparameterization trick

To backprop through sampling, write $z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$. The randomness is now external; the gradient flows through $\mu_\phi$ and $\sigma_\phi$ deterministically. Without this, the gradient of an expectation over a parameter-dependent distribution would require REINFORCE (high variance).

## What VAEs are good and bad at

**Good**:

- Smooth, continuous latent space useful for interpolation and editing.
- Stable training (unlike GANs).
- Good likelihood estimation (after IWAE-style correction).
- Excellent as compressors / latent encoders for downstream models (Stable Diffusion's first stage).

**Bad**:

- Image samples are **blurry** compared to GANs and diffusion. The Gaussian decoder + per-pixel MSE penalizes high-frequency detail.
- KL penalty causes **posterior collapse** in some configurations (decoder ignores $z$, output becomes nearly mean-only).
- Lower-quality samples than diffusion at the same parameter count.

## Common pitfalls

- **Posterior collapse.** When the decoder is too powerful relative to the encoder, $q_\phi(z|x) \to p(z)$ and the latent becomes useless. Mitigations: KL annealing, free bits, reduce decoder capacity, β-VAE with $\beta < 1$ early in training.
- **Forgetting the reparameterization trick.** Sampling $z$ inside the network and trying to backprop through `z = sample(N(mu, sigma))` doesn't work; use `z = mu + sigma * epsilon`.
- **Treating ELBO as the model's likelihood.** ELBO is a lower bound; for likelihood comparison use IWAE estimates.
- **Using VAEs as competitive standalone image generators.** They aren't anymore; use them as latent compressors with diffusion / AR on top.

## Variants

- **β-VAE**: scale the KL term by $\beta$. Higher $\beta$ encourages disentanglement; lower $\beta$ improves reconstruction.
- **VQ-VAE**: discrete (categorical) latent via vector quantization. Used in language-image models, audio.
- **IWAE** (importance-weighted): tighter ELBO via $K$-sample importance weighting.
- **NVAE, Hierarchical VAEs**: deep hierarchical latents for higher-fidelity generation.

## Related

- [Reparameterization trick](/questions/reparameterization-trick/). The gradient enabler.
- [Autoregressive vs. diffusion](/concepts/autoregressive-vs-diffusion/). Alternative generative paradigms.
