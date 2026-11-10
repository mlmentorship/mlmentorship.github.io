---
title: "Learning rate schedules: warmup and cosine decay"
description: "Why almost every modern training run linearly warms up the LR over a few hundred steps and then decays it on a cosine to near zero."
date: "2026-01-05"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A learning-rate schedule is a function $\eta(t)$ that varies the optimizer's step size over training. The dominant 2026 default for LLMs is **linear warmup** for a few hundred to a few thousand steps, followed by **cosine decay** down to ~10% of the peak LR.

## Why warmup

Early training can combine poorly estimated optimizer moments, high curvature near initialization, and rapidly changing activations. Jumping immediately to the target LR can make those transient updates unstable. Warmup limits the initial step size while the model and optimizer state enter a regime that can support the peak LR.

Typical warmup: $W = 1000$ to $4000$ steps for pretraining; ~100 steps is enough for fine-tuning.

## Why cosine

After warmup, decay from the peak LR toward a lower floor over the remaining horizon. Cosine decay,

$$
\eta(t) = \eta_{\min} + \tfrac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\!\left(\pi \cdot \tfrac{t - W}{T - W}\right)\right),
$$

starts and ends with a shallow slope while changing fastest near the middle of the decay interval [(Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1608.03983).

Common practice: cosine to $\eta_{\min} = 0.1 \cdot \eta_{\max}$ over the full training horizon $T$.

<!-- visual:warmup-cosine-three-checkpoints -->
<figure class="learning-figure plot-panel" aria-labelledby="warmup-cosine-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="warmup-cosine-title">A nonzero decay floor shifts the midpoint above half the peak</p><div class="visual-argument" data-visual-argument="warmup-cosine-three-checkpoints" data-argument-kind="curve" data-argument-composition="fixed-floor"><svg viewBox="0 0 360 613" role="img" aria-label="A nonzero floor lifts the cosine midpoint. After warmup, a cosine from 1 to 0.2 reaches 0.6 at its midpoint, not 0.5. The schedule is a chosen control, not a fitted learning curve."><title>A nonzero floor lifts the cosine midpoint</title><desc>After warmup, a cosine from 1 to 0.2 reaches 0.6 at its midpoint, not 0.5. The schedule is a chosen control, not a fitted learning curve.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Keep the floor; decay only the</tspan><tspan x="20" dy="18">remaining headroom</tspan></text><rect x="34" y="320" width="64" height="42" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect><rect x="34" y="152" width="64" height="168" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><text x="66" y="411" style="font-size:23px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="66" dy="0">1</tspan></text><rect x="143" y="320" width="64" height="42" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect><rect x="143" y="236" width="64" height="84" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><text x="175" y="411" style="font-size:23px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="175" dy="0">0.6</tspan></text><rect x="252" y="320" width="64" height="42" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect><rect x="252" y="320" width="64" height="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><text x="284" y="411" style="font-size:23px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="284" dy="0">0.2</tspan></text><text x="20" y="489" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Midpoint keeps the floor and half the</tspan><tspan x="20" dy="18">headroom: 0.6</tspan></text><text x="20" y="555" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">A chosen schedule, not an observed</tspan><tspan x="20" dy="18">learning curve.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 286" role="img" aria-labelledby="warmup-cosine-svg-title warmup-cosine-svg-desc">
		<title id="warmup-cosine-svg-title">Linear warmup followed by cosine decay to ten percent of peak</title>
		<desc id="warmup-cosine-svg-desc">A normalized learning-rate plot rises linearly from zero to the peak at warmup end W. It then follows a cosine curve over the separate interval from W to training end T. A circle marks one hundred percent of peak at W, a diamond marks fifty-five percent halfway between W and T, and a square marks the ten-percent floor at T. Labels, marker shapes, and guide lines identify every checkpoint without relying on color.</desc>
		<rect class="viz-plot-bg" x="48" y="32" width="288" height="188" rx="3"></rect>
		<path class="viz-gridline" d="M48 50H336M48 126.5H336M48 203H336"></path>
		<path class="viz-axis" d="M48 28V220H340"></path>
		<path class="viz-baseline" d="M48 203H336"></path>
		<path class="viz-operating-guide" d="M96 32V220M212 32V220M328 32V220"></path>
		<path d="M56 220L96 50" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-linecap:round"></path>
		<polyline points="96,50 110.5,51.5 125,55.8 139.5,62.9 154,72.4 168.5,84 183,97.2 197.5,111.6 212,126.5 226.5,141.4 241,155.8 255.5,169 270,180.6 284.5,190.1 299,197.2 313.5,201.5 328,203" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round;stroke-linejoin:round"></polyline>
		<circle cx="96" cy="50" r="5" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></circle>
		<path d="M212 120.5L218 126.5L212 132.5L206 126.5Z" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2.5"></path>
		<rect x="323" y="198" width="10" height="10" style="fill:var(--viz-neutral-bg);stroke:var(--viz-input-stroke);stroke-width:2.5"></rect>
		<text class="viz-axis-label" x="70" y="20" text-anchor="middle">LINEAR WARMUP</text>
		<text class="viz-axis-label" x="220" y="20" text-anchor="middle">COSINE OVER W → T</text>
		<text class="viz-callout" x="104" y="45">peak: 1.00 ηmax</text>
		<text class="viz-callout" x="320" y="119" text-anchor="end">midpoint: 0.55 ηmax</text>
		<text class="viz-callout" x="320" y="192" text-anchor="end">floor: 0.10 ηmax</text>
		<text class="viz-label" x="42" y="54" text-anchor="end">1.00</text>
		<text class="viz-label" x="42" y="130.5" text-anchor="end">0.55</text>
		<text class="viz-label" x="42" y="207" text-anchor="end">0.10</text>
		<text class="viz-label" x="56" y="238" text-anchor="middle">start</text>
		<text class="viz-label" x="96" y="238" text-anchor="middle">W</text>
		<text class="viz-label" x="212" y="238" text-anchor="middle">(W + T) / 2</text>
		<text class="viz-label" x="328" y="238" text-anchor="middle">T</text>
		<text class="viz-axis-label" x="192" y="273" text-anchor="middle">training step →</text>
		<text class="viz-axis-label" transform="translate(14 184) rotate(-90)">learning rate / ηmax</text>
	</svg>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> follow the line to the circle at W, where warmup reaches the peak. Only then does the cosine fraction start: halfway from W to T, the diamond is at 0.1 + ½(1 − 0.1) = 0.55 of peak. At T, the square reaches the 0.10 floor. The three marker shapes and labels carry the meaning independently of color. Original normalized plot checked against the <a href="https://arxiv.org/abs/1608.03983">SGDR cosine equation</a> and <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html">PyTorch scheduler documentation</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="warmup-cosine-three-checkpoints"><strong>Check:</strong> Why is the decay midpoint 0.55 of the peak instead of 0.50?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="warmup-cosine-three-checkpoints">After warmup, a cosine from 1 to 0.2 reaches 0.6 at its midpoint, not 0.5. The schedule is a chosen control, not a fitted learning curve. <strong>Why is the post-warmup midpoint 0.6 rather than 0.5?</strong></span></figcaption>
</figure>

## How to set the peak LR

For Adam/AdamW on transformer training, the default starting point is $\eta_{\max} = 3 \times 10^{-4}$ (Karpathy's "magic constant") for moderate batch sizes. Larger batches scale up roughly linearly until the LR-batch tradeoff breaks ($\sim 10^{-3}$ for very large batch).

For SFT or task-specific fine-tuning of a pretrained model: 10–100× lower than pretraining ($1 \times 10^{-5}$ to $5 \times 10^{-5}$).

A **LR range test** [(Smith, 2017)](https://arxiv.org/abs/1506.01186): sweep $\eta$ from $10^{-7}$ to $10^{-1}$ over a few hundred steps, plot loss vs. LR. Pick the point at the steepest descent (typically ~10× below where the loss diverges).

## Other schedules

- **Constant**: useful for online learning / RL where the data distribution shifts.
- **Inverse square root** (original transformer paper): $\eta(t) \propto 1/\sqrt{t}$ after warmup. Largely superseded by cosine.
- **One-cycle** [(Smith, 2018)](https://arxiv.org/abs/1803.09820): warmup to a high peak, then decay aggressively. Used in some vision training; uncommon in LLMs.
- **WSD** (Warmup-Stable-Decay): warmup, hold constant for most of training, decay sharply at the end. Used in some recent LLM training [(Hu et al., 2024)](https://arxiv.org/abs/2404.06395) for easier checkpoint resumption.

## Common pitfalls

- **Skipping warmup.** Adam without warmup on a transformer routinely diverges.
- **Decaying too fast.** Aggressive decay limits how far the model can move; cosine-to-10% is a reliable default.
- **Forgetting LR scales with batch.** Doubling the batch usually requires roughly doubling the LR.
- **Resuming a cosine schedule from a checkpoint.** If the cosine is parameterized over total steps, resuming with a different total breaks the schedule. Save schedule state with the checkpoint.
