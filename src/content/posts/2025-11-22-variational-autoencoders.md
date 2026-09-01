---
title: "Variational autoencoders (VAE)"
description: "Encode inputs to a latent distribution, decode samples back, optimize evidence lower bound. The cleanest gateway to deep generative models."
date: "2025-11-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **variational autoencoder** [(Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114) is a generative model with a latent variable $z$, learned encoder $q_\phi(z \mid x)$, and decoder $p_\theta(x \mid z)$, trained to maximize the **evidence lower bound** (ELBO):

$$
\log p_\theta(x) \ge \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)] - \mathrm{KL}(q_\phi(z \mid x) \,\|\, p(z)).
$$

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

**Learning objective:** trace the training path from an observed input through the approximate posterior and both ELBO terms, then contrast it with generation, which samples the prior and uses only the decoder.

<!-- visual:vae-training-and-generation-paths -->
<figure class="learning-figure plot-panel" aria-labelledby="vae-paths-title">
	<p class="visual-kicker">One model, two execution paths</p>
	<p class="visual-title" id="vae-paths-title">When does a VAE use the encoder, and when does it sample the prior?</p>
	<svg viewBox="0 0 360 490" role="img" aria-labelledby="vae-paths-svg-title vae-paths-svg-desc">
		<title id="vae-paths-svg-title">VAE training and generation paths</title>
		<desc id="vae-paths-svg-desc">In the upper training panel, observed x enters the encoder, which produces approximate posterior q phi. A reparameterized latent z flows into the decoder and the reconstruction term. Beside that path, the KL term compares q phi with prior p of z. In the lower generation panel, a dashed path samples z directly from the prior, passes it through the decoder, and produces a distribution for a new observation. The encoder and training input do not participate in generation.</desc>
		<defs>
			<marker id="vae-solid-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
			<marker id="vae-dashed-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="5" y="5" width="350" height="314" rx="6"></rect>
		<text class="viz-axis-label" x="18" y="28">TRAIN: INFER z FROM THIS x</text>
		<rect class="viz-node viz-node--input" x="18" y="48" width="72" height="46" rx="5"></rect>
		<text class="viz-node-label" x="54" y="68">x</text>
		<text class="viz-node-value" x="54" y="85">observed</text>
		<rect class="viz-node" x="111" y="48" width="82" height="46" rx="5"></rect>
		<text class="viz-node-label" x="152" y="68">encoder</text>
		<text class="viz-node-value" x="152" y="85">parameters φ</text>
		<rect class="viz-node viz-node--focus" x="214" y="48" width="128" height="46" rx="5"></rect>
		<text class="viz-node-label" x="278" y="68">q<tspan baseline-shift="sub" font-size="9">φ</tspan>(z|x)</text>
		<text class="viz-node-value" x="278" y="85">μ<tspan baseline-shift="sub" font-size="8">φ</tspan>(x), σ<tspan baseline-shift="sub" font-size="8">φ</tspan>(x)</text>
		<path d="M90 71H107" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<path d="M193 71H210" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<rect class="viz-node" x="18" y="132" width="82" height="46" rx="5" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-node-label" x="59" y="152">p(z)</text>
		<text class="viz-node-value" x="59" y="169">prior N(0,I)</text>
		<rect class="viz-node viz-node--focus" x="118" y="132" width="76" height="46" rx="5" style="stroke-dasharray:6 4"></rect>
		<text class="viz-node-label" x="156" y="152">KL</text>
		<text class="viz-node-value" x="156" y="169">regularizer</text>
		<rect class="viz-node viz-node--focus" x="230" y="132" width="96" height="46" rx="5"></rect>
		<text class="viz-node-label" x="278" y="152">z = μ + σε</text>
		<text class="viz-node-value" x="278" y="169">ε ∼ N(0,I)</text>
		<path d="M278 94V128" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<path d="M214 78H205V126H194" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.8;marker-end:url(#vae-solid-arrow)"></path>
		<path d="M100 155H114" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.8;marker-end:url(#vae-solid-arrow)"></path>
		<text class="viz-label" x="156" y="118" text-anchor="middle">compare q<tspan baseline-shift="sub" font-size="8">φ</tspan> to p</text>
		<rect class="viz-node" x="230" y="216" width="96" height="46" rx="5"></rect>
		<text class="viz-node-label" x="278" y="236">decoder</text>
		<text class="viz-node-value" x="278" y="253">parameters θ</text>
		<path d="M278 178V212" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<rect class="viz-node viz-node--output" x="145" y="279" width="197" height="28" rx="14"></rect>
		<text class="viz-node-value" x="243.5" y="297">reconstruction: log p<tspan baseline-shift="sub" font-size="8">θ</tspan>(x|z)</text>
		<path d="M278 262V275" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<path d="M90 82V293H141" style="fill:none;stroke:var(--viz-edge);stroke-width:1.4;stroke-dasharray:4 4;marker-end:url(#vae-solid-arrow)"></path>
		<text class="viz-label" x="25" y="200">target x</text>
		<text class="viz-callout" x="18" y="304">ELBO = reconstruction - KL</text>
		<rect class="viz-plot-bg" x="5" y="335" width="350" height="150" rx="6"></rect>
		<text class="viz-axis-label" x="18" y="358">GENERATE: SAMPLE z WITHOUT AN INPUT x</text>
		<rect class="viz-node" x="18" y="384" width="82" height="46" rx="5" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-node-label" x="59" y="404">p(z)</text>
		<text class="viz-node-value" x="59" y="421">sample z</text>
		<rect class="viz-node" x="139" y="384" width="82" height="46" rx="5"></rect>
		<text class="viz-node-label" x="180" y="404">decoder</text>
		<text class="viz-node-value" x="180" y="421">same θ</text>
		<rect class="viz-node viz-node--output" x="260" y="384" width="82" height="46" rx="5"></rect>
		<text class="viz-node-label" x="301" y="404">p<tspan baseline-shift="sub" font-size="9">θ</tspan>(x|z)</text>
		<text class="viz-node-value" x="301" y="421">new x</text>
		<path d="M100 407H135" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#vae-dashed-arrow)"></path>
		<path d="M221 407H256" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#vae-dashed-arrow)"></path>
		<text class="viz-label" x="180" y="461" text-anchor="middle">No observed x, encoder, q<tspan baseline-shift="sub" font-size="8">φ</tspan>, or KL step.</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the solid training path first: observed <var>x</var> determines <var>q</var><sub>φ</sub>, a reparameterized sample <var>z</var> reaches the decoder, reconstruction scores how well the decoder explains that same <var>x</var>, and KL pulls <var>q</var><sub>φ</sub> toward the prior. Then follow the dashed generation path: sample <var>z</var> directly from <var>p</var>(<var>z</var>) and run only the decoder to produce a new observation. Original schematic checked against <a href="https://arxiv.org/abs/1312.6114">Kingma and Welling</a>, <a href="https://arxiv.org/abs/1401.4082">Rezende et al.</a>, and <a href="https://arxiv.org/abs/1906.02691">the VAE tutorial</a>.</figcaption>
</figure>

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
