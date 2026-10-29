---
title: "Generative adversarial networks (GANs)"
description: "Two networks compete: a generator produces samples, a discriminator distinguishes them from real data. Sharp samples, training instability, mostly displaced by diffusion in 2026."
date: "2026-01-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **GAN** [(Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661) trains two networks adversarially: a **generator** $G$ that maps noise $z \sim p_z$ to samples $G(z)$, and a **discriminator** $D$ that tries to distinguish $G(z)$ from real samples $x \sim p_\text{data}$. The minimax objective:

$$
\min_G \max_D \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].
$$

GANs produced the sharpest, most realistic image samples of the deep learning era from 2015 to 2021, peaking with StyleGAN3 and BigGAN. They have largely been displaced by diffusion for image generation in 2026, but remain relevant in:

- Real-time / latency-critical generation (single forward pass vs diffusion's iterative).
- Image-to-image translation (CycleGAN, pix2pix).
- Specialized domains (medical, super-resolution).
- As a discriminator-style critic in other systems (perceptual losses, adversarial robustness).

Knowing GAN training dynamics is also key to understanding why diffusion's stable training is such an advantage.

## The two players

- **Generator $G$**: maps noise $z$ to a fake sample. Trained to fool $D$.
- **Discriminator $D$**: binary classifier distinguishing real from fake. Trained to maximize correct classification.

<p class="visual-kicker">Learning objective</p>
<!-- visual:gan-alternating-update-ownership -->
<figure class="learning-figure editorial-illustration" aria-labelledby="gan-alternating-update-ownership-edition-title"><p class="visual-title" id="gan-alternating-update-ownership-edition-title">Frozen discriminator weights must still pass gradients to the generator</p><svg viewBox="0 0 360 430" role="img" aria-labelledby="gan-alternating-update-ownership-edition-svg-title gan-alternating-update-ownership-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="gan-alternating-update-ownership-edition-svg-title">Freezing discriminator parameters must not detach the generator's gradient path</title><desc id="gan-alternating-update-ownership-edition-svg-desc">The discriminator step detaches generated samples, updating D but not G. The generator step freezes D parameters while a dashed loss-gradient path passes through D back into G. The two panels distinguish no parameter update from no gradient path.</desc><text x="180" y="23" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">1. DISCRIMINATOR STEP</tspan></text><g data-editorial-node><rect x="18" y="54" width="126" height="67" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="81" y="73.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="81" dy="0">G produces</tspan><tspan x="81" dy="19">fake</tspan><tspan x="81" dy="19">G NOT updated</tspan></text></g><g data-editorial-node><rect x="216" y="54" width="126" height="67" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="279" y="73.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="279" dy="0">D sees real +</tspan><tspan x="279" dy="19">fake</tspan><tspan x="279" dy="19">UPDATE D</tspan></text></g><path d="M144 88L216 88" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M209.8569220667674,91.35597877022943L216,88L209.8569220667674,84.64402122977057" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="145" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">fake is detached: no gradient back into G</tspan></text><path d="M18 180L342 180" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="223" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">2. GENERATOR STEP</tspan></text><g data-editorial-node><rect x="18" y="252" width="126" height="67" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="81" y="271.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="81" dy="0">G produces</tspan><tspan x="81" dy="19">fake</tspan><tspan x="81" dy="19">UPDATE G</tspan></text></g><g data-editorial-node><rect x="216" y="252" width="126" height="67" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="279" y="271.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="279" dy="0">D evaluates</tspan><tspan x="279" dy="19">fake</tspan><tspan x="279" dy="19">D NOT updated</tspan></text></g><path d="M144 285L216 285" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M209.8569220667674,288.3559787702294L216,285L209.8569220667674,281.6440212297706" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M278 319L278 349" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M278 349L81 349" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M81 349L81 319" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M84.35597877022943,325.14307793323263L81,319L77.64402122977057,325.14307793323263" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="180" y="383" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Gradient passes through D into G.</tspan></text><text x="180" y="411" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Frozen parameters != a detached computation</tspan></text></svg><figcaption><strong>Read it this way:</strong> The discriminator step detaches G's output. The generator step instead differentiates through frozen D. No parameter update and no gradient path are different operations. <span class="visual-reader-check" data-reader-check="gan-alternating-update-ownership"><strong>Check:</strong> Why must gradients pass through frozen D during the generator step?</span></figcaption></figure>

At equilibrium (Nash), $G$ produces samples indistinguishable from real, and $D$ outputs $\tfrac{1}{2}$ everywhere.

## Why training is hard

The minimax game is unstable for many reasons:

1. **Mode collapse**: $G$ finds one or a few outputs that consistently fool $D$ and ignores the rest of the distribution.
2. **Vanishing gradients**: when $D$ is much better than $G$, $\nabla_G \log(1 - D(G(z))) \to 0$. No learning signal.
3. **Non-convergence**: minimax dynamics can cycle without converging to equilibrium.
4. **Sensitivity to architecture and hyperparameters**: small changes make a working GAN diverge.

A decade of research produced **many stabilization techniques**: spectral normalization, two-time-scale updates (TTUR), gradient penalty, WGAN/WGAN-GP, R1 regularization, progressive growing, StyleGAN's mapping network. Each helps; none fully solves it.

## Variants

| GAN | Innovation |
|-----|-----------|
| DCGAN [(Radford 2015)](https://arxiv.org/abs/1511.06434) | Convolutional architecture for images |
| WGAN [(Arjovsky 2017)](https://arxiv.org/abs/1701.07875) | Wasserstein loss; weight clipping or gradient penalty (WGAN-GP) for Lipschitz constraint |
| Conditional GAN | Add class label or text embedding to both $G$ and $D$ |
| pix2pix, CycleGAN | Image-to-image translation (paired and unpaired) |
| BigGAN [(Brock 2018)](https://arxiv.org/abs/1809.11096) | Class-conditional ImageNet generation at scale |
| StyleGAN 1/2/3 ([Karras 2018](https://arxiv.org/abs/1812.04948)-2021) | Mapping network + AdaIN + alias-free design; SoTA face generation |

## Why diffusion replaced GANs for image generation

| Property | GAN | Diffusion |
|----------|-----|-----------|
| Training stability | Notoriously unstable | Stable |
| Sample quality (FID) | Excellent | Excellent (better at scale) |
| Mode coverage | Mode collapse risk | Better coverage |
| Sample speed | One forward pass (fast) | Many denoising steps (slow) |
| Likelihood | None | Variational lower bound |
| Conditioning flexibility | Limited | Cross-attention conditioning is strong |

Diffusion's training stability and conditioning flexibility (especially text-to-image with classifier-free guidance) tipped the balance.

## Where GANs still win in 2026

- **Real-time generation**: single forward pass beats diffusion's tens of steps.
- **Style transfer / image-to-image**: CycleGAN-style pipelines remain strong.
- **Adversarial training as a regularizer**: not for generation per se, but as a critic loss in distillation, super-resolution, domain adaptation.

## Common pitfalls

- **Treating GAN inception/FID scores as the only metric.** They miss diversity issues; complement with precision/recall or coverage metrics.
- **Not using spectral normalization or gradient penalty.** Vanilla GAN training without modern stabilization is very fragile.
- **Comparing GAN samples to diffusion samples at matched compute without matching steps.** GAN: 1 forward pass; diffusion: 50–1000. Per sample, GAN is much cheaper.
- **Reading "GANs are mode collapsed" as universal.** Modern StyleGAN-class models cover ImageNet diversity well.

## Related

- [Autoregressive vs. diffusion](/concepts/autoregressive-vs-diffusion/). Broader generative paradigm map.
- [Variational autoencoders](/concepts/variational-autoencoders/). Earlier alternative.
