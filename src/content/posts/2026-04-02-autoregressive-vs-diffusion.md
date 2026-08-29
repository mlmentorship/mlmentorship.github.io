---
title: "Autoregressive vs. diffusion generation"
description: "Two paradigms for generative modeling: predict the next element step-by-step (autoregressive) or iteratively denoise from pure noise (diffusion). Different costs, different strengths."
date: "2026-04-02"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Autoregressive (AR)** models factorize $p(x) = \prod_t p(x_t \mid x_{<t})$ and generate one element at a time. **Diffusion** models learn to invert a Markov noising process and generate by iteratively denoising from Gaussian noise. AR dominates language; diffusion dominates images.

The two paradigms produce very different production tradeoffs:

| Aspect | Autoregressive | Diffusion |
|--------|---------------|-----------|
| Sampling | $T$ sequential steps (one per token) | $S$ sequential steps (~10–1000 denoise steps) |
| Parallelism within sample | None during generation | Full (one denoise step is parallel) |
| Quality scaling | Compute and data | Compute and data + step count |
| Modality strength | Discrete sequences (text, code) | Continuous (images, audio) |
| Conditioning | Prefix prompt | Cross-attention or classifier-free guidance |
| Likelihood | Exact, easy to compute | Variational lower bound; sample-based |

## Autoregressive

The model factorizes the joint distribution by the chain rule:

$$
p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}).
$$

A neural net (transformer, RNN) parameterizes each $p(x_t \mid x_{<t})$. Training: maximize log-likelihood = minimize cross-entropy (one prediction per token, all positions in parallel via teacher forcing). Sampling: feed back the previous output, generate the next.

**Strengths**: exact likelihood, simple training, parallel teacher-forced loss, strong on discrete sequences.

**Weakness**: serial sampling. Each step waits for the previous. This is the bottleneck that motivates [speculative decoding](/concepts/speculative-decoding/).

## Diffusion

Define a forward noising process: $x_0 \to x_1 \to \dots \to x_T$ with $x_T \sim \mathcal{N}(0, I)$. The forward step is a small Gaussian (variance schedule $\beta_t$). The reverse (denoising) step is approximated by a learned model $\epsilon_\theta(x_t, t)$:

$$
x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I).
$$

Training: sample a clean $x_0$, sample noise $\epsilon$, sample $t$, compute noisy $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$, fit $\epsilon_\theta(x_t, t) \approx \epsilon$ with MSE.

Sampling: $x_T \sim \mathcal{N}(0, I)$, then iterate the reverse step $T$ times (typically 10–1000).

**Strengths**: high-fidelity continuous generation, stable training (no GAN instability), good likelihood estimates with importance-weighted ELBO.

**Weakness**: slow sampling. Most efforts (DDIM, distillation, consistency models) reduce step count.

## When AR vs. diffusion

| Modality | Production default 2026 |
|----------|------------------------|
| Text | Autoregressive (Llama, GPT, Mistral) |
| Code | Autoregressive (GPT-4, Codex) |
| Images | Diffusion (Stable Diffusion, FLUX, Imagen) |
| Audio | Mixed: AR (WaveNet legacy), diffusion (modern TTS), latent autoregressive |
| Video | Diffusion (Sora, Veo, Stable Video) with latent compression |
| Molecules / proteins | Diffusion (RFdiffusion) |

For text, AR has structural advantages: discrete vocabulary, natural causal ordering, and chain-of-thought reasoning emerges from sequential generation. For images, no natural sequential ordering exists, and diffusion's iterative refinement maps better to gradual denoising.

## Hybrid and emerging approaches

- **Latent diffusion** (Stable Diffusion): VAE compresses to latent space; diffusion happens there.
- **Discrete diffusion** for text (D3PM, Plaid): apply diffusion to discrete tokens with a categorical noise process.
- **Flow matching** [(Meta, 2023)](https://arxiv.org/abs/2210.02747): generalizes diffusion to a deterministic ODE; faster sampling.
- **Consistency models** [(Song 2023)](https://arxiv.org/abs/2303.01469): one or few-step diffusion via distillation.
- **Diffusion language models** [(LLaDA, 2024)](https://arxiv.org/abs/2502.09992): diffusion applied to text; competitive with AR at smaller scale, scaling unclear.

## Common pitfalls

- **Comparing AR perplexity with diffusion ELBO directly.** Different objectives; not directly comparable.
- **Treating diffusion step count as fixed.** It is a sample-time hyperparameter; reducing improves throughput at quality cost.
- **Forgetting AR is parallel during training.** The "AR is slow" concern applies to inference, not training.
- **Assuming diffusion is universally better than GANs.** It is for image fidelity and training stability, not necessarily for inference speed; GANs still dominate latency-critical settings.
