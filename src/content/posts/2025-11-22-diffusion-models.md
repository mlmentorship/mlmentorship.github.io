---
title: "Diffusion models"
description: "Learn to invert a fixed noising process. The dominant generative paradigm for images, audio, video, and molecules in 2026."
date: "2025-11-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A **diffusion model** ([Ho et al., 2020](https://arxiv.org/abs/2006.11239); [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)) defines a forward Markov chain that gradually adds Gaussian noise to data $x_0 \to x_1 \to \dots \to x_T \approx \mathcal{N}(0, I)$, and learns a neural network $\epsilon_\theta(x_t, t)$ to reverse it by predicting the noise added at each step. Sampling iterates the learned reverse process from pure noise.

## Why it matters

Diffusion is the **dominant 2026 paradigm** for high-fidelity generation in continuous modalities:

- Images: Stable Diffusion, DALL-E 3, Midjourney, Imagen, FLUX.
- Video: Sora, Veo, Runway Gen-3.
- Audio: Stable Audio, AudioLDM, Suno.
- Molecules / proteins: RFdiffusion [(Watson, Juergens, Bennett et al., 2023)](https://www.nature.com/articles/s41586-023-06415-8) for protein structure generation; widely used in the lab of David Baker, who shared the 2024 Chemistry Nobel for computational protein design.

Compared to GANs and VAEs, diffusion offers stable training, excellent sample diversity, and natural conditional generation. Its main weakness. Slow iterative sampling. Is the focus of active research (DDIM, distillation, consistency models).

## The forward process

Define a variance schedule $\beta_1, \dots, \beta_T$ (small values, increasing). The forward step:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I).
$$

This is a fixed (non-learned) Markov chain. After $T$ steps with appropriate schedule, $x_T \approx \mathcal{N}(0, I)$.

A useful identity: you can sample $x_t$ directly from $x_0$ in closed form:

$$
q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0,\; (1 - \bar\alpha_t) I), \quad \bar\alpha_t = \prod_{s=1}^{t}(1 - \beta_s).
$$

So $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ for $\epsilon \sim \mathcal{N}(0, I)$.

## Training

For a sample $x_0$:

1. Pick a random timestep $t \in \{1, \dots, T\}$.
2. Sample noise $\epsilon \sim \mathcal{N}(0, I)$.
3. Form $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$.
4. Train $\epsilon_\theta(x_t, t) \approx \epsilon$ with MSE.

That's it. One simple loss, no adversary, no special tricks. This is why diffusion is so stable. Denoising regression on a closed-form forward process.

## Sampling (reverse process)

DDPM-style ancestral sampling [(Ho et al., 2020)](https://arxiv.org/abs/2006.11239):

1. Start with $x_T \sim \mathcal{N}(0, I)$.
2. For $t = T, T-1, \dots, 1$:
   - Compute predicted noise $\epsilon_\theta(x_t, t)$.
   - Sample $z \sim \mathcal{N}(0, I)$ (or $z = 0$ at $t = 1$).
   - Update $x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(x_t, t)) + \sigma_t z$.
3. Return $x_0$.

DDPM uses $T = 1000$. **DDIM** [(Song et al., 2020)](https://arxiv.org/abs/2010.02502) reduces to ~50 steps with a deterministic update. **Consistency models** [(Song 2023)](https://arxiv.org/abs/2303.01469) and **distillation** further reduce to 1–4 steps.

## Conditional generation

Add condition $c$ (text embedding, class label, image) to the network: $\epsilon_\theta(x_t, t, c)$. Standard conditioning mechanism: cross-attention from $x_t$ to text embeddings (T5 or CLIP).

**Classifier-free guidance** [(Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598) interpolates between conditional and unconditional predictions:

$$
\tilde\epsilon_\theta(x_t, t, c) = (1 + w) \epsilon_\theta(x_t, t, c) - w \epsilon_\theta(x_t, t, \emptyset)
$$

with $w \approx 7$ for text-to-image. Trades sample diversity for adherence to the prompt.

## Latent diffusion (Stable Diffusion)

Train a VAE to compress images to a small latent space (typically 1/8 spatial resolution, 4 channels). Run diffusion in the latent space. Decode the final latent with the VAE decoder.

Result: 64×64 latent diffusion ≈ 512×512 image quality at much lower compute. The 2022 SD release (Rombach et al.) made high-quality text-to-image practical on consumer GPUs.

## Sample quality vs. step count

Diffusion sample quality is determined by:

- Model capacity and training data.
- Number of denoising steps (more = better, with diminishing returns).
- Sampler (DDIM, DPM-Solver, Euler, Heun). Different ODE/SDE solvers with different quality/speed tradeoffs.
- Classifier-free guidance scale.

A useful rule: 30–50 DDIM steps with a modern sampler matches 1000-step DDPM quality.

## Common pitfalls

- **Confusing $\epsilon_\theta$ prediction with $x_0$ prediction.** Diffusion can be parameterized as predicting the noise, the clean image, or the velocity (v-parameterization). They are mathematically related but not identical.
- **Forgetting that the forward process is fixed, not learned.** Only the reverse is parameterized.
- **Treating diffusion as a likelihood model directly.** Diffusion has a variational ELBO; for likelihood-based comparison, use IWAE or compare on FID/sample quality instead.
- **Running fewer steps than the model was trained for, without testing.** Some samplers degrade past 10 steps; check empirically.

## Related

- [Variational autoencoders](/concepts/variational-autoencoders/). VAEs are the latent compressor in latent diffusion.
- [Autoregressive vs. diffusion](/concepts/autoregressive-vs-diffusion/). Broader paradigm comparison.
