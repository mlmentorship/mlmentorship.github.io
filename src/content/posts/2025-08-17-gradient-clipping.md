---
title: "Gradient clipping"
description: "Cap the norm of the gradient before each optimizer step. The simplest and most reliable defense against training instability."
date: "2025-08-17"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Gradient clipping rescales the gradient vector before the optimizer step so that its global norm does not exceed a fixed threshold $c$. If $\|g\| > c$, replace $g$ with $g \cdot c / \|g\|$; otherwise leave it unchanged.

<!-- visual:global-norm-clipping-radial-projection -->
<figure class="learning-figure" aria-labelledby="gradient-clipping-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="gradient-clipping-title">See why global-norm clipping shortens a gradient without turning it</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 250" role="img" aria-labelledby="gradient-clipping-svg-title gradient-clipping-svg-desc">
			<title id="gradient-clipping-svg-title">Global-norm clipping as a radial projection</title>
			<desc id="gradient-clipping-svg-desc">A dashed original gradient g equals 6 comma 8 and extends from the origin beyond a circular norm limit. Its norm is 10 while the threshold c is 5. A solid clipped gradient follows the same ray and stops halfway at the circle, at 3 comma 4 with norm 5. Direct labels, different endpoint shapes, and solid versus dashed lines distinguish the vectors without color.</desc>
			<circle cx="85" cy="166" r="70" style="fill:var(--viz-neutral-bg);stroke:var(--viz-edge);stroke-width:1.5"></circle>
			<path class="viz-gridline" d="M15 166H180M85 236V28"></path>
			<path class="viz-baseline" d="M85 166L169 54"></path>
			<path class="viz-pr-curve" d="M85 166L127 110"></path>
			<path d="M169 47L175 59H163Z" style="fill:var(--viz-neutral-bg);stroke:var(--viz-edge);stroke-width:1.5"></path>
			<rect x="122" y="105" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<circle cx="85" cy="166" r="3" style="fill:var(--c-text-soft)"></circle>
			<text class="viz-label" x="78" y="182" text-anchor="end">origin</text>
			<text class="viz-axis-label" x="26" y="102">norm limit c = 5</text>
			<text class="viz-callout" x="139" y="103">clipped g′</text>
			<text class="viz-label" x="177" y="48">original g</text>
			<path d="M195 25V225" style="stroke:var(--c-rule);stroke-width:1"></path>
			<text class="viz-axis-label" x="214" y="57">ORIGINAL</text>
			<text class="viz-callout" x="214" y="78">g = (6, 8)</text>
			<text class="viz-label" x="214" y="97">‖g‖₂ = 10</text>
			<text class="viz-axis-label" x="214" y="132">SCALE EVERY COMPONENT</text>
			<text class="viz-callout" x="214" y="153">c / ‖g‖₂ = 5 / 10</text>
			<text class="viz-axis-label" x="214" y="188">RESULT</text>
			<text class="viz-callout" x="214" y="209">g′ = (3, 4)</text>
			<text class="viz-label" x="214" y="228">‖g′‖₂ = 5 · same ray</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> begin with the dashed gradient <var>g</var>, whose norm 10 lies outside the radius-5 limit. Multiplying every component by the same factor, 5/10, moves its endpoint radially to the circle at (3, 4). The solid clipped vector is shorter but stays on the same ray, so global-norm clipping preserves direction; clipping components independently need not. The dashed line, square and triangle endpoints, positions, and direct labels carry the explanation without color. Original schematic based on <a href="https://proceedings.mlr.press/v28/pascanu13.html">Pascanu, Mikolov, and Bengio (2013)</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html">PyTorch global-norm definition</a>.</figcaption>
</figure>

Training instabilities. Loss spikes, NaN gradients, exploding updates. Are usually caused by a single batch with anomalous gradients. Without clipping, that one bad step can drive parameters into a region from which training never recovers. Clipping bounds the worst-case update and turns a divergence into a recoverable hiccup.

Standard in: transformer pretraining (always), most RL training (always), RNN training (originally proposed for RNNs by [Pascanu et al., 2013](https://arxiv.org/abs/1211.5063), where exploding gradients are intrinsic).

## Two flavors

### Global-norm clipping (the standard)

Compute the L2 norm of the concatenated gradient vector across **all** parameters:

$$
\|g\|_2 = \sqrt{\sum_p \|g_p\|_2^2}
$$

If $\|g\|_2 > c$, scale every parameter's gradient by $c / \|g\|_2$. This preserves the *direction* of the gradient (just shrinks magnitude). $c = 1.0$ is the dominant default for transformer training.

### Per-parameter clipping

Clip each parameter's gradient norm independently. Simpler but distorts the gradient direction; rarely used.

### Value clipping

Clip individual elements of $g$ to a range $[-c, +c]$. Distorts direction even more; mostly historical.

## How to pick the threshold

- **Transformers**: $c = 1.0$ is the universal default. Llama, Mistral, Qwen, GPT all use 1.0.
- **RNNs / LSTMs**: $c$ between 0.25 and 5; needs tuning.
- **RL**: depends on reward scale and policy parametrization; often $c = 0.5$.
- **Diagnostic**: log $\|g\|_2$ over training. If it almost never exceeds $c$, the clip is inactive (try lower); if it always does, the clip is destroying signal (try higher).

## Combined with mixed precision

In FP16/BF16 mixed precision, the loss is scaled before the backward pass to keep small gradients representable. Clipping must be applied **on the unscaled gradients** (after the scaler unscales them). PyTorch's `GradScaler` and similar tooling enforce this ordering.

## Common pitfalls

- **Clipping per parameter group instead of globally.** Gives a different effective clip for each parameter; rarely intended.
- **Forgetting to unscale before clipping under AMP.** The clip threshold is meaningless if applied to scaled gradients.
- **Setting clipping too aggressive.** $c = 0.01$ for a transformer cripples training; you'll see flat loss curves with the clip always active.
- **Treating clipping as a fix for a buggy data pipeline.** A consistent stream of large gradients usually indicates a data or initialization problem, not a clipping problem.
