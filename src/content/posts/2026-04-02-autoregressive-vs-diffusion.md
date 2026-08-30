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

<!-- visual:autoregressive-diffusion-sampling-steps -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="ar-diffusion-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ar-diffusion-visual-title">At one serial sampling step, which parts of the state can change?</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 410" role="img" aria-labelledby="ar-diffusion-svg-title ar-diffusion-svg-desc">
			<title id="ar-diffusion-svg-title">Autoregressive appending versus diffusion whole-state revision</title>
			<desc id="ar-diffusion-svg-desc">Two aligned sampling timelines each contain four positions. On the autoregressive timeline, A is given, then each serial model call appends exactly one new element while earlier elements remain fixed. On the diffusion timeline, a complete four-position noise state exists from the start, and each serial denoising call revises all four positions together until a clean sample is reached.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="18" y="27">AUTOREGRESSIVE: append one element; keep the prefix fixed</text>
			<text class="viz-label" x="18" y="53">given prefix</text>
			<rect class="viz-node viz-node--input" x="18" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="32" y="88">A</text>
			<rect class="viz-node" x="49" y="64" width="28" height="38" rx="4" stroke-dasharray="4 3"></rect><text class="viz-node-value" x="63" y="87">?</text>
			<rect class="viz-node" x="80" y="64" width="28" height="38" rx="4" stroke-dasharray="4 3"></rect><text class="viz-node-value" x="94" y="87">?</text>
			<rect class="viz-node" x="111" y="64" width="28" height="38" rx="4" stroke-dasharray="4 3"></rect><text class="viz-node-value" x="125" y="87">?</text>
			<path class="viz-forward" d="M151 83 H202"></path><text class="viz-edge-label" x="176" y="70">serial call 1</text>
			<text class="viz-label" x="214" y="53">append B</text>
			<rect class="viz-node" x="214" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="228" y="88">A</text>
			<rect class="viz-node viz-node--focus" x="245" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="259" y="88">B</text>
			<rect class="viz-node" x="276" y="64" width="28" height="38" rx="4" stroke-dasharray="4 3"></rect><text class="viz-node-value" x="290" y="87">?</text>
			<rect class="viz-node" x="307" y="64" width="28" height="38" rx="4" stroke-dasharray="4 3"></rect><text class="viz-node-value" x="321" y="87">?</text>
			<path class="viz-forward" d="M347 83 H398"></path><text class="viz-edge-label" x="372" y="70">serial call 2</text>
			<text class="viz-label" x="410" y="53">append C</text>
			<rect class="viz-node" x="410" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="424" y="88">A</text>
			<rect class="viz-node" x="441" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="455" y="88">B</text>
			<rect class="viz-node viz-node--focus" x="472" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="486" y="88">C</text>
			<rect class="viz-node" x="503" y="64" width="28" height="38" rx="4" stroke-dasharray="4 3"></rect><text class="viz-node-value" x="517" y="87">?</text>
			<path class="viz-forward" d="M543 83 H594"></path><text class="viz-edge-label" x="568" y="70">serial call 3</text>
			<text class="viz-label" x="606" y="53">append D</text>
			<rect class="viz-node" x="606" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="620" y="88">A</text>
			<rect class="viz-node" x="637" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="651" y="88">B</text>
			<rect class="viz-node" x="668" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="682" y="88">C</text>
			<rect class="viz-node viz-node--focus" x="699" y="64" width="28" height="38" rx="4"></rect><text class="viz-node-label" x="713" y="88">D</text>
			<path d="M245 111 V121 H273 V111" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2"></path><text class="viz-edge-label" x="259" y="137">one new element</text>
			<path d="M472 111 V121 H500 V111" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2"></path><text class="viz-edge-label" x="486" y="137">one new element</text>
			<path d="M699 111 V121 H727 V111" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2"></path><text class="viz-edge-label" x="713" y="137">one new element</text>
			<line class="viz-gridline" x1="18" y1="168" x2="742" y2="168"></line>
			<text class="viz-axis-label" x="18" y="203">DIFFUSION: revise the entire current state within each denoising step</text>
			<text class="viz-label" x="18" y="229">x<tspan baseline-shift="sub" font-size="8">T</tspan>: full noise state</text>
			<rect class="viz-node viz-node--input" x="18" y="240" width="121" height="52" rx="8"></rect>
			<text class="viz-node-label" x="33" y="270">·</text><text class="viz-node-label" x="63" y="270">×</text><text class="viz-node-label" x="94" y="270">~</text><text class="viz-node-label" x="124" y="270">·</text>
			<path class="viz-forward" d="M151 266 H202"></path><text class="viz-edge-label" x="176" y="253">serial call 1</text>
			<text class="viz-label" x="214" y="229">x<tspan baseline-shift="sub" font-size="8">T−1</tspan>: less noisy</text>
			<rect class="viz-node viz-node--focus" x="214" y="240" width="121" height="52" rx="8"></rect>
			<text class="viz-node-label" x="229" y="270">╱</text><text class="viz-node-label" x="259" y="270">○</text><text class="viz-node-label" x="290" y="270">△</text><text class="viz-node-label" x="320" y="270">╲</text>
			<path class="viz-forward" d="M347 266 H398"></path><text class="viz-edge-label" x="372" y="253">serial call 2</text>
			<text class="viz-label" x="410" y="229">x<tspan baseline-shift="sub" font-size="8">T−2</tspan>: structure emerges</text>
			<rect class="viz-node viz-node--focus" x="410" y="240" width="121" height="52" rx="8"></rect>
			<text class="viz-node-label" x="425" y="270">A</text><text class="viz-node-label" x="455" y="270">B</text><text class="viz-node-label" x="486" y="270">△</text><text class="viz-node-label" x="516" y="270">D</text>
			<path class="viz-forward" d="M543 266 H594"></path><text class="viz-edge-label" x="568" y="253">… serial calls</text>
			<text class="viz-label" x="606" y="229">x<tspan baseline-shift="sub" font-size="8">0</tspan>: clean sample</text>
			<rect class="viz-node viz-node--output" x="606" y="240" width="121" height="52" rx="8"></rect>
			<text class="viz-node-label" x="621" y="270">A</text><text class="viz-node-label" x="651" y="270">B</text><text class="viz-node-label" x="682" y="270">C</text><text class="viz-node-label" x="712" y="270">D</text>
			<path d="M214 302 V314 H335 V302" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2"></path><text class="viz-edge-label" x="274" y="331">all positions revised together</text>
			<path d="M410 302 V314 H531 V302" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2"></path><text class="viz-edge-label" x="470" y="331">all positions revised together</text>
			<text class="viz-callout" x="380" y="375" text-anchor="middle">Both timelines are serial across calls; only diffusion parallelizes work across positions inside a call.</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> scan left to right along both rows. An autoregressive call freezes the existing prefix and adds one element, so the three new elements shown require three dependent calls. A diffusion call consumes and revises the whole noisy state at once, but the next denoising call must still wait for that complete state. Mechanism checked against <a href="https://arxiv.org/abs/1601.06759">PixelRNN</a> and <a href="https://arxiv.org/abs/2006.11239">DDPM</a>; the schematic is original.</figcaption>
</figure>

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
