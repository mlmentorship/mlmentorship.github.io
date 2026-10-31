---
title: "Gradient clipping"
description: "Project a finite global gradient onto a norm ball, then distinguish that bound from the optimizer's final parameter displacement."
date: "2025-08-17"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Global-norm clipping shortens a finite gradient by one common factor, preserving its direction when the factor is positive.
For threshold $c\ge0$, use $g'=g\min(1,c/\|g\|_2)$, leaving a zero vector unchanged.
This caps the gradient norm, not every optimizer's eventual update.

The fixture has $g=(6,8)$, norm ten, and threshold five.
Scaling by one-half produces $(3,4)$ with norm five.
Ordinary SGD with learning rate $0.1$ moves by $(-0.3,-0.4)$, norm $0.5$.

<!-- visual:global-norm-clipping-radial-projection -->
<figure class="learning-figure" aria-labelledby="gradient-clipping-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="gradient-clipping-title">Shorten the whole gradient vector without changing its direction</p><div class="visual-argument" data-visual-argument="global-norm-clipping-radial-projection" data-argument-kind="geometry" data-argument-composition="vectors"><svg viewBox="0 0 360 443" role="img" aria-label="Shorten the vector without turning it. The vector (3,4) has norm 5. Clipping to norm 2 multiplies every component by 2/5, giving (1.2,1.6)."><title>Shorten the vector without turning it</title><desc>The vector (3,4) has norm 5. Clipping to norm 2 multiplies every component by 2/5, giving (1.2,1.6).</desc><g opacity=".3"><path d="M318 36L318 312" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M42 36L318 36" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".3"><path d="M272 36L272 312" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M42 82L318 82" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".3"><path d="M226 36L226 312" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M42 128L318 128" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".3"><path d="M180 36L180 312" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M42 174L318 174" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".3"><path d="M134 36L134 312" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M42 220L318 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".3"><path d="M42 36L42 312" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M42 312L318 312" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M32 266L330 266" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M88 26L88 322" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="328" y="255" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="328" dy="0">x</tspan></text><text x="98" y="25" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="98" dy="0">y</tspan></text><path d="M42 263L42 269" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="42" y="286" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="42" dy="0">-1</tspan></text><path d="M134 263L134 269" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="134" y="286" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="134" dy="0">1</tspan></text><path d="M180 263L180 269" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="286" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="180" dy="0">2</tspan></text><path d="M226 263L226 269" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="226" y="286" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="226" dy="0">3</tspan></text><path d="M272 263L272 269" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="272" y="286" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="272" dy="0">4</tspan></text><path d="M318 263L318 269" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="318" y="286" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="318" dy="0">5</tspan></text><path d="M88 266L226 82" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3.5"></path><path d="M224.2693374928964 90.83203302113992L226 82" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M218.00583380169468 86.13440525273863L226 82" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-proof-vector="0" data-proof-x="3" data-proof-y="4"><circle cx="226" cy="82" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M88 266L143.2 192.39999999999998" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2.5;stroke-dasharray:7 3"></path><path d="M141.4693374928964 201.2320330211399L143.2 192.39999999999998" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M135.20583380169467 196.53440525273862L143.2 192.39999999999998" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><g data-proof-vector="1" data-proof-x="1.2" data-proof-y="1.6"><rect x="139.2" y="188.39999999999998" width="8" height="8" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect></g><path d="M20 349L45 349" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle cx="32" cy="349" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="57" y="353" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">Original gradient; norm 5: (3, 4)</tspan></text><path d="M20 381L45 381" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><rect x="28" y="377" width="8" height="8" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="57" y="385" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">Clipped gradient; norm 2: (1.2,</tspan><tspan x="57" dy="18">1.6)</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> the dashed gradient (6, 8) has norm ten. A common factor one-half places (3, 4) on the radius-five circle. The solid vector retains its direction. The optimizer still transforms it before changing parameters. Original schematic follows <a href="https://proceedings.mlr.press/v28/pascanu13.html">Pascanu, Mikolov, and Bengio (2013)</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html">PyTorch global-norm definition</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="global-norm-clipping-radial-projection"><strong>Check:</strong> Why does global clipping produce (3,4), rather than independently truncating both coordinates to 5?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="global-norm-clipping-radial-projection">The vector (3,4) has norm 5. Clipping to norm 2 multiplies every component by 2/5, giving (1.2,1.6).</span></figcaption>
</figure>

Among vectors inside the radius-$c$ Euclidean ball, radial projection minimizes distance to $g$.
Cauchy-Schwarz makes the nearest nonzero candidate parallel to $g$; its permitted length is $\min(\|g\|,c)$.
This geometric guarantee neither diagnoses a loss spike nor makes a nonfinite gradient usable.
Check finiteness before stepping; infinity multiplied by zero does not produce a repaired direction.

## Two flavors

### Global-norm clipping (the standard)

Compute the norm over the declared optimizer parameter set after its intended accumulation and synchronization:

$$
\|g\|_2 = \sqrt{\sum_p \|g_p\|_2^2}
$$

Use that same factor for every included tensor.
For sharded gradients, aggregate squared norms over unique shards; replicated data-parallel copies are not additional parameter coordinates.

### Per-parameter clipping

Separate tensor caps change relative scaling and generally rotate the concatenated gradient.

### Value clipping

Coordinate clipping to $[-5,5]$ turns $(6,8)$ into $(5,5)$, with norm $\sqrt{50}$ and a different direction.
That is a coordinate bound, not the radius-five constraint.

## How to pick the threshold

Log the preclip norm, nonfinite frequency, clipping frequency, and actual optimizer update norm.
Threshold units depend on loss reduction, parameterization, and batch accumulation.
Rare clipping may be an intentional safety limit; frequent clipping warrants checking the intended learning-rate and loss-scale interaction.
Validate a threshold against the training objective rather than treating an architecture-wide number as universal.

## Combined with mixed precision

If loss scaling is used, finish accumulation and unscale once before clipping.
Otherwise the threshold applies to scaled units; dividing afterward changes the effective cap.
BF16 frequently runs without a loss scaler; format choice alone does not imply one.

## Change the Requirement

### Bound Parameter Movement

**New task:** Cap the final optimizer displacement rather than its input gradient.

**Picture and change:** Inspect momentum, preconditioning, learning rate, and decay before choosing an update-level constraint.

**Check:** With velocity $v=(100,0)$ and rule $v'=0.9v+g'$, the fixture gives $(93,4)$, not a norm-five update.

**Use when:** The requirement concerns actual parameter movement.
The bound $\|\Delta\theta\|\le\eta c$ applies to ordinary SGD without those additional transformations.

### Accumulate Before Clipping

**New task:** Combine several microbatches into one optimizer step under a declared summed-loss convention.

**Picture and change:** Sum contributions first, then clip the combined gradient.

**Check:** Scalar contributions $10$ and $-9$ sum to $1$; clipping each to five before adding instead gives zero.

**Use when:** Multiple backward calls represent one update.
For mean loss, apply the correct sample-count normalization; averaging unequal batch means defines a different gradient.

<!-- learning:core-end -->
<div data-learning-extension>

## Common pitfalls

- **Wrong constrained object.** Tensor, coordinate, example, gradient, and update clipping solve different problems.
- **Wrong units.** Unscale before measuring the threshold when using AMP scaling.
- **Invalid directions.** Skip or diagnose nonfinite gradients; do not expect a norm projection to repair them.
- **Missing diagnosis.** Repeated large gradients can reflect data, initialization, objective, or numerical issues; clipping alone identifies none.

## Reconstruct

<details>
<summary>Does threshold one bound every AdamW step by its learning rate?</summary>

No. Saved moments, coordinate scaling, and decoupled decay also contribute to parameter movement.
The ordinary-SGD bound cannot be transferred without inspecting the actual update rule.

</details>

</div>
