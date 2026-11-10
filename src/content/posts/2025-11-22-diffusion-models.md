---
title: "Diffusion models"
description: "Construct one noisy training pair, recover a clean estimate, and distinguish noise prediction from dependent reverse sampling."
date: "2025-11-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

Gaussian diffusion learns reverse transitions for a fixed process that corrupts clean data with noise.

## Summary

Why can training jump directly to one noise level while generation needs dependent updates?
Take clean scalar $x_0=2$, retained signal variance $\bar\alpha_t=0.64$, and sampled standard-normal noise $\epsilon=-1$.
A predictor returns $\hat\epsilon=-0.5$.
We will compute its training error and clean estimate, then identify what a sampler still needs.

<!-- visual:diffusion-training-sampling-asymmetry -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="diffusion-asymmetry-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="diffusion-asymmetry-visual-title">Corrupt once to train a noise predictor; generate through dependent reverse updates</p><div class="visual-argument" data-visual-argument="diffusion-training-sampling-asymmetry" data-argument-kind="ledger" data-argument-composition="dependent-generation"><svg viewBox="0 0 360 1725" role="img" aria-label="Corrupt once to train a noise predictor; generate through dependent reverse updates. At one sampled time, retained signal and independent noise form the input. The model sees this input and time, not target noise. Its prediction determines training error and a clean estimate. Generation instead repeatedly uses the previous state, denoiser output, sampler coefficients and any declared transition noise. A clean estimate is not a sampling update."><title>Corrupt once to train a noise predictor; generate through dependent reverse updates</title><desc>At one sampled time, retained signal and independent noise form the input. The model sees this input and time, not target noise. Its prediction determines training error and a clean estimate. Generation instead repeatedly uses the previous state, denoiser output, sampler coefficients and any declared transition noise. A clean estimate is not a sampling update.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">1. Corrupt data at one sampled time</tspan></text><text x="20" y="104" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Clean value two; retained variance</tspan><tspan x="20" dy="18">share 0.64</tspan></text><text x="20" y="182" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Retained clean signal: 1.6</tspan></text><path d="M38 231L238 231M230 235L238 231L230 227" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:4;stroke-linejoin:round"></path><path d="M238 231L238 286" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M238 286L163 286M171 282L163 286L171 290" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:4;stroke-linejoin:round"></path><text x="20" y="345" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Independent noise contribution: minus</tspan><tspan x="20" dy="18">0.6</tspan></text><path d="M38 410L288 410" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M38 404L38 416" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="38" y="447" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="38" dy="0">0</tspan></text><path d="M163 404L163 416" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="163" y="447" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="163" dy="0">1</tspan></text><path d="M288 404L288 416" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="288" y="447" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="288" dy="0">2</tspan></text><circle cx="163" cy="410" r="7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="20" y="493" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Model input: noisy value one and</tspan><tspan x="20" dy="18">sampled time</tspan></text><text x="20" y="585" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">2. Compare predicted noise with its</tspan><tspan x="20" dy="18">target</tspan></text><text x="20" y="657" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Training-only target: minus one</tspan></text><path d="M60 715L300 715" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="60" cy="715" r="8" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="180" cy="715" r="7" style="fill:var(--viz-state-stroke);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="60" y="753" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="60" dy="0">-1</tspan></text><text x="300" y="753" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="300" dy="0">0</tspan></text><path d="M60 784L180 784" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><path d="M60 777L60 791" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><path d="M180 777L180 791" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><text x="20" y="838" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="20" dy="0">Predicted noise: minus one-half.</tspan><tspan x="20" dy="18">Squared error: 0.25.</tspan></text><text x="20" y="941" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">3. Undo the same corruption to</tspan><tspan x="20" dy="18">estimate clean data</tspan></text><text x="20" y="1019" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="20" dy="0">Predicted noise gives clean estimate</tspan><tspan x="20" dy="18">1.625</tspan></text><path d="M38 1090L288 1090" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="241.125" cy="1090" r="7" style="fill:var(--viz-state-stroke);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><circle cx="288" cy="1090" r="8" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="38" y="1128" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="38" dy="0">0</tspan></text><text x="288" y="1128" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="288" dy="0">2</tspan></text><text x="20" y="1186" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">The true noise would recover the</tspan><tspan x="20" dy="18">original two.</tspan></text><text x="20" y="1293" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">4. Generate through dependent reverse</tspan><tspan x="20" dy="18">states</tspan></text><text x="20" y="1370" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Illustrative states at times three,</tspan><tspan x="20" dy="18">two, one, zero</tspan></text><circle cx="50" cy="1455" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="50" y="1461" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">3</tspan></text><circle cx="134" cy="1455" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="134" y="1461" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="134" dy="0">2</tspan></text><path d="M78 1455L106 1455M98 1459L106 1455L98 1451" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="218" cy="1455" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="218" y="1461" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">1</tspan></text><path d="M162 1455L190 1455M182 1459L190 1455L182 1451" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="302" cy="1455" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="302" y="1461" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="302" dy="0">0</tspan></text><path d="M246 1455L274 1455M266 1459L274 1455L266 1451" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="1532" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Each next state needs its</tspan><tspan x="20" dy="18">predecessor, denoiser output and</tspan><tspan x="20" dy="18">declared sampler coefficients.</tspan></text><text x="20" y="1640" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Transition noise depends on the</tspan><tspan x="20" dy="18">sampler. A clean estimate alone is</tspan><tspan x="20" dy="18">not the next state.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
			<text class="viz-node-label" x="68" y="91">x<tspan baseline-shift="sub" font-size="10">0</tspan></text>
			<text class="viz-node-value" x="68" y="111">clean sample</text>
			<text class="viz-label" x="148" y="54">sample t and ε</text>
			<rect class="viz-node" x="143" y="65" width="112" height="58" rx="7"></rect>
			<text class="viz-node-label" x="199" y="91">t ~ Uniform</text>
			<text class="viz-node-value" x="199" y="111">ε ~ N(0, I)</text>
			<path d="M118 94 H130 V136 H276 V112 H296" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#diffusion-arrow-forward)"></path>
			<path d="M255 94 H296" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#diffusion-arrow-forward)"></path>
			<text class="viz-edge-label" x="276" y="82">direct jump</text>
			<rect class="viz-node viz-node--focus" x="296" y="65" width="166" height="58" rx="7"></rect>
			<text class="viz-node-label" x="379" y="89">x<tspan baseline-shift="sub" font-size="10">t</tspan> in closed form</text>
			<text class="viz-node-value" x="379" y="110">√ᾱ<tspan baseline-shift="sub" font-size="10">t</tspan>x<tspan baseline-shift="sub" font-size="10">0</tspan> + √(1−ᾱ<tspan baseline-shift="sub" font-size="10">t</tspan>)ε</text>
			<path d="M462 94 H500" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-forward)"></path>
			<text class="viz-edge-label" x="481" y="82">one call</text>
			<rect class="viz-node viz-node--focus" x="500" y="65" width="116" height="58" rx="7"></rect>
			<text class="viz-node-label" x="558" y="89">ε<tspan baseline-shift="sub" font-size="10">θ</tspan>(x<tspan baseline-shift="sub" font-size="10">t</tspan>, t)</text>
			<text class="viz-node-value" x="558" y="111">predict noise ε̂</text>
			<path d="M616 94 H642" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#diffusion-arrow-forward)"></path>
			<rect class="viz-node viz-node--output" x="642" y="65" width="100" height="58" rx="7"></rect>
			<text class="viz-node-label" x="692" y="90">MSE(ε, ε̂)</text>
			<text class="viz-node-value" x="692" y="111">update θ</text>
			<path d="M199 123 V145 H692 V123" fill="none" stroke="var(--viz-edge)" stroke-width="1.4" stroke-dasharray="5 4"></path>
			<text class="viz-edge-label" x="445" y="160">the sampled ε is the available target; no forward or reverse chain is unrolled</text>
			<line class="viz-gridline" x1="18" y1="184" x2="742" y2="184"></line>
			<text class="viz-axis-label" x="18" y="216">SAMPLING: each computed state is input to the next reverse step</text>
			<text class="viz-label" x="18" y="242">start from noise</text>
			<rect class="viz-node viz-node--input" x="18" y="253" width="100" height="58" rx="7"></rect>
			<text class="viz-node-label" x="68" y="278">x<tspan baseline-shift="sub" font-size="10">T</tspan></text>
			<text class="viz-node-value" x="68" y="299">N(0, I)</text>
			<path d="M118 282 H151" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-backward)"></path>
			<rect class="viz-node viz-node--focus" x="151" y="253" width="112" height="58" rx="7"></rect>
			<text class="viz-node-label" x="207" y="278">ε<tspan baseline-shift="sub" font-size="10">θ</tspan>(x<tspan baseline-shift="sub" font-size="10">T</tspan>, T)</text>
			<text class="viz-node-value" x="207" y="299">predict noise ε̂</text>
			<path d="M263 282 H296" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-backward)"></path>
			<text class="viz-edge-label" x="279" y="270">update</text>
			<rect class="viz-node" x="296" y="253" width="100" height="58" rx="7"></rect>
			<text class="viz-node-label" x="346" y="278">x<tspan baseline-shift="sub" font-size="10">T−1</tspan></text>
			<text class="viz-node-value" x="346" y="299">computed state</text>
			<path d="M396 282 H429" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-backward)"></path>
			<rect class="viz-node viz-node--focus" x="429" y="253" width="112" height="58" rx="7"></rect>
			<text class="viz-node-label" x="485" y="278">ε<tspan baseline-shift="sub" font-size="10">θ</tspan>(x<tspan baseline-shift="sub" font-size="10">t</tspan>, t)</text>
			<text class="viz-node-value" x="485" y="299">predict noise ε̂</text>
			<path d="M541 282 H574" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2.4" marker-end="url(#diffusion-arrow-backward)"></path>
			<text class="viz-edge-label" x="558" y="270">…</text>
			<rect class="viz-node viz-node--output" x="574" y="253" width="168" height="58" rx="7"></rect>
			<text class="viz-node-label" x="658" y="278">x<tspan baseline-shift="sub" font-size="10">0</tspan></text>
			<text class="viz-node-value" x="658" y="299">generated clean sample</text>
			<path d="M151 321 V337 H541 V321" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="2"></path>
			<text class="viz-gradient-label" x="346" y="354">many dependent predictor calls: x<tspan baseline-shift="sub" font-size="10">t−1</tspan> must exist before the next call</text>
			<text class="viz-callout" x="380" y="389" text-anchor="middle">One shared time-conditioned network learns across noise levels; only generation chains its calls.</text>
		</svg>
	</div>
	<div class="visual-derivation-caption">compare the arrows in the two lanes. Training knows clean data and sampled noise, so one marginal draw supplies a target. Generation knows neither target. Each noise prediction feeds a separate reverse update, whose result becomes the next input. Skipping levels requires compatible solver coefficients. Sources: <a href="https://arxiv.org/abs/2006.11239">DDPM</a> and <a href="https://arxiv.org/abs/2010.02502">DDIM</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="diffusion-training-sampling-asymmetry"><strong>Check:</strong> Why does training need one sampled noise level while generation must wait for each preceding state?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="diffusion-training-sampling-asymmetry">At one sampled time, retained signal and independent noise form the input. The model sees this input and time, not target noise. Its prediction determines training error and a clean estimate. Generation instead repeatedly uses the previous state, denoiser output, sampler coefficients and any declared transition noise. A clean estimate is not a sampling update. <strong>Would exact noise recover two, and would that alone define the next state?</strong></span></figcaption>
</figure>

## The forward process

Choose $0<\beta_t<1$, define $\alpha_t=1-\beta_t$, and let $\bar\alpha_t=\prod_{s=1}^t\alpha_s$.
Independent forward Gaussian increments give

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I),
\qquad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

Composition multiplies the signal coefficients.
The noise variance recurs as $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$.
Thus the marginal is $\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$.
It approaches standard normal only when residual signal becomes negligible for the data distribution.

## Training

Sample data, a time index and independent $\epsilon\sim\mathcal N(0,I)$.
Construct $x_t$ directly and minimize $\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2$.
This simplified objective need not equal every time-weighted variational bound.
Our fixture gives $x_t=0.8(2)+0.6(-1)=1$ and squared error $(-1+0.5)^2=0.25$.
The model sees $x_t,t$, not the sampled target noise.

**Change the requirement:** predict clean data instead.
For $0<\bar\alpha_t<1$, convert using

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}}.
$$

The estimate is $(1+0.3)/0.8=1.625$; using the true noise would recover two.
Noise MSE equals clean-data MSE multiplied by $\bar\alpha_t/(1-\bar\alpha_t)$ under this conversion.
Changing prediction targets without their loss weights and sampler coefficients changes the training problem.
At zero retained signal, this inverse formula is singular.

## Sampling (reverse process)

Start from an appropriate Gaussian prior and compute dependent reverse states:

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)+\sigma_t z.
$$

Here $z$ is fresh standard-normal transition noise, with $z=0$ at the final step by convention.
The variance $\sigma_t^2$ is a declared model or sampler choice.
One choice is $\beta_t(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$; it is not the only supported variance.
A noise prediction alone supplies neither the update coefficients nor the next state.
Using adjacent-step DDPM coefficients for an arbitrary timestep jump is not justified.

## Conditional generation

**Change the requirement:** strengthen a condition $c$ using a model trained to support unconditional prediction.
[Classifier-free guidance](https://arxiv.org/abs/2207.12598) combines

$$
{}\tilde\epsilon=(1+w)\epsilon_\theta(x_t,t,c)-w\epsilon_\theta(x_t,t,\emptyset).
$$

Conditional prediction two, unconditional prediction one and $w=2$ produce $3(2)-2(1)=4$.
The alternative convention $s=1+w$ writes unconditional plus $s$ times the conditional difference.
Guidance changes the sampled distribution; it is not simply a speed setting.

**Transfer:** ten transitions with two denoiser evaluations each require twenty evaluations, absent reuse.
Batching the pair changes execution scheduling, not the number of logical evaluations.

<!-- learning:core-end -->
<div data-learning-extension>

## Latent diffusion (Stable Diffusion)

An additional learned encoder maps data to a smaller latent; diffusion models that latent distribution.
The decoder maps generated latents back to the data space.
[Latent diffusion](https://arxiv.org/abs/2112.10752) changes the representation and reconstruction limits, not the Gaussian composition identity.
Compression ratio and channel count depend on the autoencoder.
Text cross-attention and protein conditioning, as in [RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8), are distinct application choices.

## Sample quality vs. step count

Measure quality, coverage, latency and denoiser evaluations for the actual model, schedule, guidance and solver.
More evaluations do not universally improve every metric.
[DDIM](https://arxiv.org/abs/2010.02502), [consistency models](https://arxiv.org/abs/2303.01469) and distillation require compatible updates or learned objectives.
Their reported speedups are not universal equivalence rules.

## Common pitfalls

Likelihood bounds and likelihood estimates evaluate probability assignment under stated approximations.
FID evaluates feature-distribution statistics; it is not a likelihood estimate or a complete coverage test.
Noise, clean-data and velocity parameterizations require matching time-dependent conversions.
The forward corruption here is fixed; the learned reverse remains an approximation.
Foundations: [DDPM](https://arxiv.org/abs/2006.11239) and [Sohl-Dickstein et al.](https://arxiv.org/abs/1503.03585).

## Related

- [Variational autoencoders](/concepts/variational-autoencoders/). VAEs are the latent compressor in latent diffusion.
- [Autoregressive vs. diffusion](/concepts/autoregressive-vs-diffusion/). Broader paradigm comparison.

</div>
