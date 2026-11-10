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
	<p class="visual-title" id="gradient-clipping-title">Cap the finite global gradient along one ray; do not confuse it with an optimizer-step bound</p><div class="visual-argument" data-visual-argument="global-norm-clipping-radial-projection" data-argument-kind="geometry" data-argument-composition="gradient-norm-and-step-contract"><svg viewBox="0 0 360 2030" role="img" aria-label="Cap the finite global gradient along one ray; do not confuse it with an optimizer-step bound. Global clipping scales every included gradient by one factor, preserving direction and capping its norm. This fixture becomes three and four; SGD moves by norm 0.5. Value clipping gives two fives and changes direction. Aggregate unique parameter coordinates, not replicated copies. Finish accumulation and synchronization, unscale, check finiteness, then clip before stepping. Adaptive optimizers need a separate update-norm check. Clipping does not repair nonfinite gradients."><title>Cap the finite global gradient along one ray; do not confuse it with an optimizer-step bound</title><desc>Global clipping scales every included gradient by one factor, preserving direction and capping its norm. This fixture becomes three and four; SGD moves by norm 0.5. Value clipping gives two fives and changes direction. Aggregate unique parameter coordinates, not replicated copies. Finish accumulation and synchronization, unscale, check finiteness, then clip before stepping. Adaptive optimizers need a separate update-norm check. Clipping does not repair nonfinite gradients.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">One common factor caps the global</tspan><tspan x="20" dy="18">gradient norm</tspan></text><text x="20" y="110" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Gradient components six and eight;</tspan><tspan x="20" dy="18">norm ten; threshold five</tspan></text><path d="M45 350L330 350" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 160L180 485" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="180" cy="350" r="75" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-dasharray:4 4"></circle><path d="M180 350L270 230M268.4 238.8L270 230L262 234" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><path d="M270 230L225 290M226.60000000000002 281.20000000000005L225 290L233 286" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-linejoin:round"></path><path d="M180 350L225 290M223.39999999999998 298.79999999999995L225 290L217 294" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="270" cy="230" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="225" cy="290" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="255" cy="275" r="6" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="20" y="560" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Global clip: components three and</tspan><tspan x="20" dy="18">four; norm five, same ray</tspan></text><text x="20" y="665" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Value clip: two fives, norm about</tspan><tspan x="20" dy="18">7.07; outside the same ball and a</tspan><tspan x="20" dy="18">different direction</tspan></text><text x="20" y="790" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">SGD at learning rate 0.1:</tspan><tspan x="20" dy="18">displacement components minus 0.3 and</tspan><tspan x="20" dy="18">minus 0.4; norm 0.5.</tspan></text><text x="20" y="900" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Adaptive or momentum updates can have</tspan><tspan x="20" dy="18">a different displacement; inspect the</tspan><tspan x="20" dy="18">actual update norm.</tspan></text><text x="20" y="1020" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Finish accumulation and</tspan><tspan x="20" dy="18">synchronization. Aggregate unique</tspan><tspan x="20" dy="18">parameter shards; replicas do not add</tspan><tspan x="20" dy="18">coordinates.</tspan></text><circle cx="50" cy="1190" r="22" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="50" y="1196" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">1</tspan></text><text x="100" y="1184" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="100" dy="0">Unscale once</tspan></text><path d="M50 1221L50 1300M46 1292L50 1300L54 1292" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="50" cy="1340" r="22" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="50" y="1346" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">2</tspan></text><text x="100" y="1334" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="100" dy="0">Reject nonfinite gradients</tspan></text><path d="M50 1371L50 1450M46 1442L50 1450L54 1442" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="50" cy="1490" r="22" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="50" y="1496" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">3</tspan></text><text x="100" y="1484" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="100" dy="0">Global-norm clip</tspan></text><path d="M50 1521L50 1600M46 1592L50 1600L54 1592" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="50" cy="1640" r="22" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="50" y="1646" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">4</tspan></text><text x="100" y="1634" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="100" dy="0">Optimizer step</tspan></text><text x="20" y="1770" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Scale one hundred: clipping before</tspan><tspan x="20" dy="18">unscale would leave norm 0.05, not</tspan><tspan x="20" dy="18">five.</tspan></text><text x="20" y="1890" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A zero vector stays zero. Clipping is</tspan><tspan x="20" dy="18">a finite-gradient constraint, not a</tspan><tspan x="20" dy="18">remedy for infinity or a diagnosis of</tspan><tspan x="20" dy="18">a loss spike.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> the dashed gradient (6, 8) has norm ten. A common factor one-half places (3, 4) on the radius-five circle. The solid vector retains its direction. The optimizer still transforms it before changing parameters. Original schematic follows <a href="https://proceedings.mlr.press/v28/pascanu13.html">Pascanu, Mikolov, and Bengio (2013)</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html">PyTorch global-norm definition</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="global-norm-clipping-radial-projection"><strong>Check:</strong> Why does global clipping produce (3,4), rather than independently truncating both coordinates to 5?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="global-norm-clipping-radial-projection">Global clipping scales every included gradient by one factor, preserving direction and capping its norm. This fixture becomes three and four; SGD moves by norm 0.5. Value clipping gives two fives and changes direction. Aggregate unique parameter coordinates, not replicated copies. Finish accumulation and synchronization, unscale, check finiteness, then clip before stepping. Adaptive optimizers need a separate update-norm check. Clipping does not repair nonfinite gradients. <strong>Does value clipping satisfy the same radius-five constraint?</strong></span></figcaption>
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
