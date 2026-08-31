---
title: "Diffusion models"
description: "Learn to invert a fixed noising process. The dominant generative paradigm for images, audio, video, and molecules in 2026."
date: "2025-11-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **diffusion model** ([Ho et al., 2020](https://arxiv.org/abs/2006.11239); [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)) defines a forward Markov chain that gradually adds Gaussian noise to data $x_0 \to x_1 \to \dots \to x_T \approx \mathcal{N}(0, I)$, and learns a neural network $\epsilon_\theta(x_t, t)$ to reverse it by predicting the noise added at each step. Sampling iterates the learned reverse process from pure noise.

Diffusion is the **dominant 2026 paradigm** for high-fidelity generation in continuous modalities:

- Images: Stable Diffusion, DALL-E 3, Midjourney, Imagen, FLUX.
- Video: Sora, Veo, Runway Gen-3.
- Audio: Stable Audio, AudioLDM, Suno.
- Molecules / proteins: RFdiffusion [(Watson, Juergens, Bennett et al., 2023)](https://www.nature.com/articles/s41586-023-06415-8) for protein structure generation; widely used in the lab of David Baker, who shared the 2024 Chemistry Nobel for computational protein design.

Compared to GANs and VAEs, diffusion offers stable training, excellent sample diversity, and natural conditional generation. Its main weakness. Slow iterative sampling. Is the focus of active research (DDIM, distillation, consistency models).

**Learning objective:** Distinguish one-step diffusion training at an independently sampled noise level from sampling, which must chain learned reverse transitions from noise to a clean sample.

<!-- visual:diffusion-training-sampling-asymmetry -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="diffusion-asymmetry-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="diffusion-asymmetry-visual-title">Why can training jump to one noise level while sampling must walk backward?</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 410" role="img" aria-labelledby="diffusion-asymmetry-svg-title diffusion-asymmetry-svg-desc">
			<title id="diffusion-asymmetry-svg-title">Diffusion training uses one sampled timestep while generation chains reverse steps</title>
			<desc id="diffusion-asymmetry-svg-desc">The training lane starts with clean data x zero and sampled Gaussian noise epsilon. A closed-form equation jumps directly to an independently selected noisy state x t without constructing earlier states. One call to the time-conditioned network predicts epsilon, and its error against the known sampled noise trains the shared parameters. The sampling lane starts from Gaussian noise x T. It calls the same time-conditioned network repeatedly; each reverse update produces x t minus one, which is required before the next call, until a clean sample x zero is produced.</desc>
			<defs>
				<marker id="diffusion-arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="diffusion-arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="18" y="28">TRAINING: choose one independent noise level per example</text>
			<text class="viz-label" x="18" y="54">known clean data</text>
			<rect class="viz-node viz-node--input" x="18" y="65" width="100" height="58" rx="7"></rect>
			<text class="viz-node-label" x="68" y="91">x<tspan baseline-shift="sub" font-size="9">0</tspan></text>
			<text class="viz-node-value" x="68" y="111">clean sample</text>
			<text class="viz-label" x="148" y="54">sample t and ε</text>
			<rect class="viz-node" x="143" y="65" width="112" height="58" rx="7"></rect>
			<text class="viz-node-label" x="199" y="91">t ~ Uniform</text>
			<text class="viz-node-value" x="199" y="111">ε ~ N(0, I)</text>
			<path d="M118 94 H143" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#diffusion-arrow-forward)"></path>
			<path d="M255 94 H296" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#diffusion-arrow-forward)"></path>
			<text class="viz-edge-label" x="276" y="82">direct jump</text>
			<rect class="viz-node viz-node--focus" x="296" y="65" width="166" height="58" rx="7"></rect>
			<text class="viz-node-label" x="379" y="89">x<tspan baseline-shift="sub" font-size="9">t</tspan> in closed form</text>
			<text class="viz-node-value" x="379" y="110">√ᾱ<tspan baseline-shift="sub" font-size="8">t</tspan>x<tspan baseline-shift="sub" font-size="8">0</tspan> + √(1−ᾱ<tspan baseline-shift="sub" font-size="8">t</tspan>)ε</text>
			<path d="M462 94 H500" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-forward)"></path>
			<text class="viz-edge-label" x="481" y="82">one call</text>
			<rect class="viz-node viz-node--focus" x="500" y="65" width="116" height="58" rx="7"></rect>
			<text class="viz-node-label" x="558" y="89">ε<tspan baseline-shift="sub" font-size="9">θ</tspan>(x<tspan baseline-shift="sub" font-size="9">t</tspan>, t)</text>
			<text class="viz-node-value" x="558" y="111">predict noise ε̂</text>
			<path d="M616 94 H642" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#diffusion-arrow-forward)"></path>
			<rect class="viz-node viz-node--output" x="642" y="65" width="100" height="58" rx="7"></rect>
			<text class="viz-node-label" x="692" y="90">MSE(ε, ε̂)</text>
			<text class="viz-node-value" x="692" y="111">update θ</text>
			<path d="M199 123 V145 H692 V123" fill="none" stroke="var(--viz-edge)" stroke-width="1.4" stroke-dasharray="5 4"></path>
			<text class="viz-edge-label" x="445" y="160">the sampled ε is the available target; no forward or reverse chain is unrolled</text>
			<line class="viz-gridline" x1="18" y1="184" x2="742" y2="184"></line>
			<text class="viz-axis-label" x="18" y="216">SAMPLING: each predicted state is input to the next reverse step</text>
			<text class="viz-label" x="18" y="242">start from noise</text>
			<rect class="viz-node viz-node--input" x="18" y="253" width="100" height="58" rx="7"></rect>
			<text class="viz-node-label" x="68" y="278">x<tspan baseline-shift="sub" font-size="9">T</tspan></text>
			<text class="viz-node-value" x="68" y="299">N(0, I)</text>
			<path d="M118 282 H151" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-backward)"></path>
			<rect class="viz-node viz-node--focus" x="151" y="253" width="112" height="58" rx="7"></rect>
			<text class="viz-node-label" x="207" y="278">ε<tspan baseline-shift="sub" font-size="9">θ</tspan>(x<tspan baseline-shift="sub" font-size="9">T</tspan>, T)</text>
			<text class="viz-node-value" x="207" y="299">reverse update</text>
			<path d="M263 282 H296" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-backward)"></path>
			<rect class="viz-node" x="296" y="253" width="100" height="58" rx="7"></rect>
			<text class="viz-node-label" x="346" y="278">x<tspan baseline-shift="sub" font-size="9">T−1</tspan></text>
			<text class="viz-node-value" x="346" y="299">next state</text>
			<path d="M396 282 H429" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-backward)"></path>
			<rect class="viz-node viz-node--focus" x="429" y="253" width="112" height="58" rx="7"></rect>
			<text class="viz-node-label" x="485" y="278">ε<tspan baseline-shift="sub" font-size="9">θ</tspan>(x<tspan baseline-shift="sub" font-size="9">t</tspan>, t)</text>
			<text class="viz-node-value" x="485" y="299">repeat in order</text>
			<path d="M541 282 H574" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-backward)"></path>
			<text class="viz-edge-label" x="558" y="270">…</text>
			<rect class="viz-node viz-node--output" x="574" y="253" width="168" height="58" rx="7"></rect>
			<text class="viz-node-label" x="658" y="278">x<tspan baseline-shift="sub" font-size="9">0</tspan></text>
			<text class="viz-node-value" x="658" y="299">generated clean sample</text>
			<path d="M151 321 V337 H541 V321" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2"></path>
			<text class="viz-gradient-label" x="346" y="354">many dependent predictor calls: x<tspan baseline-shift="sub" font-size="8">t−1</tspan> must exist before the next call</text>
			<text class="viz-callout" x="380" y="389" text-anchor="middle">One shared time-conditioned network learns across noise levels; only generation chains its calls.</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the arrow structure, not the colors. On the top row, known clean data and sampled noise let training construct any <em>x<sub>t</sub></em> directly, supervise one network call, and update the shared parameters. On the bottom row, generation has no known clean target: every predicted <em>x<sub>t−1</sub></em> becomes the next input, so reverse calls remain serial even when a faster sampler skips selected noise levels. Mechanism checked against <a href="https://arxiv.org/abs/2006.11239">DDPM</a> and <a href="https://arxiv.org/abs/2010.02502">DDIM</a>; the schematic is original.</figcaption>
</figure>

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
