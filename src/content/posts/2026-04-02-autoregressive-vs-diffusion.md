---
title: "Autoregressive vs. diffusion generation"
description: "Two paradigms for generative modeling: predict the next element step-by-step (autoregressive) or iteratively denoise from pure noise (diffusion). Different costs, different strengths."
date: "2026-04-02"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Autoregressive sampling appends an element conditioned on the retained prefix; conventional diffusion sampling revises a complete provisional state.
Both can require dependent rounds even when a model call contains parallel computation.
Compare output support, streaming behavior, quality, and actual work, not a modality stereotype.

The two paradigms produce very different production tradeoffs:

| Aspect | Autoregressive | Diffusion |
|--------|---------------|-----------|
| State change | Append under the ordinary sampler | Revise provisional positions |
| Outer dependence | Next prefix depends on sampled elements | Next state depends on the previous transition |
| Within-call work | Architecture-dependent matrix parallelism | Architecture-dependent whole-state work |
| Likelihood | Normalized factors under the chosen representation | Bounds/estimates depend on the model and objective |

<!-- visual:autoregressive-diffusion-sampling-steps -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="ar-diffusion-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ar-diffusion-visual-title">Append one new position or revise the whole provisional sample</p>
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
			<text class="viz-callout" x="380" y="375" text-anchor="middle">Both timelines are serial across calls; diffusion revises all positions within each illustrated call.</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> scan each row from left to right. An ordinary autoregressive call appends one element. A diffusion call revises the provisional whole state. Both illustrated timelines depend on earlier calls. The original schematic follows <a href="https://arxiv.org/abs/1601.06759">PixelRNN</a> and <a href="https://arxiv.org/abs/2006.11239">DDPM</a>. <span class="visual-reader-check" data-reader-check="autoregressive-diffusion-sampling-steps"><strong>Check:</strong> Which positions can change on the second call? Why are both illustrated timelines serial?</span></figcaption>
</figure>

## Autoregressive

The model factorizes the joint distribution by the chain rule:

$$
p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}).
$$

Given prefix A, let $P(B\mid A)=0.5$, $P(C\mid AB)=0.8$, and $P(D\mid ABC)=0.25$.
Then $P(BCD\mid A)=0.1$, and ordinary sampling needs three dependent append calls.
Teacher-forced scoring knows the targets, allowing parallel positions in suitable causal-transformer implementations without revealing future targets.
Teacher forcing does not remove every architecture's sequential dependencies.

## Diffusion

Gaussian diffusion trains a reverse model using corrupted states.
Sampling begins from a chosen noise prior and applies valid reverse transitions.
A three-transition teaching sampler can revise all four positions each round, but its rounds remain dependent.
The AR product above supplies no diffusion likelihood.

### Require an Immediately Usable Prefix

**New task:** Release B before C or D exists.

**Picture and change:** The append state AB is already a valid prefix; a conventional whole-state intermediate remains provisional.

**Check:** One append releases B; one denoising transition does not generally finalize the first position.

**Use when:** Streaming is an explicit service requirement.

### Compare Actual Call Costs

**New task:** Compare three-call fixtures costing two versus five milliseconds per call.

**Picture and change:** Multiply evaluated calls by call cost; account for overhead separately.

**Check:** Totals are six versus fifteen milliseconds despite equal round counts.

**Use when:** Quality is separately qualified; these invented costs are not a performance benchmark.

<!-- learning:core-end -->
<div data-learning-extension>

For a Gaussian schedule, $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s\le t}\alpha_s$.
Training can construct $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ and fit a noise predictor by squared error.
A conventional reverse parameterization is

$$
x_{t-1}=\frac{x_t-\beta_t\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}}{\sqrt{1-\beta_t}}+\sigma_tz.
$$

Use $z\sim\mathcal N(0,I)$ with the sampler's declared variance and endpoint policy.
A finite forward schedule need not make its terminal marginal exactly standard Gaussian.

## When AR vs. diffusion

| Modality | Possible model families |
|----------|------------------------|
| Text/code | AR, discrete diffusion, hybrids |
| Images/audio/video | AR, diffusion, flows, hybrids |
| Structures | Representation-dependent generative models |

## Hybrid and emerging approaches

- **Latent diffusion:** noising in a learned representation.
- **Discrete diffusion:** categorical corruption/reconstruction, including [LLaDA](https://arxiv.org/abs/2502.09992).
- **[Flow matching](https://arxiv.org/abs/2210.02747):** learn a probability-path vector field.
- **[Consistency models](https://arxiv.org/abs/2303.01469):** train compatible few-step mappings under their specified objectives.

## Common pitfalls

- **Incompatible measures.** Probability masses, densities, and variational bounds are not interchangeable likelihood reports.
- **Invalid skipping.** Removing trained timesteps without a compatible sampler need not preserve a valid transition or quality.
- **Round-count rankings.** Work, batching, hardware, and quality matter separately.

</div>
