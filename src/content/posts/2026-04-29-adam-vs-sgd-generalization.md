---
title: "Why does Adam sometimes generalize worse than SGD?"
description: "Separate measured optimizer performance from coordinate-dependent sharpness, decay coupling and unequal tuning or resource budgets."
date: "2026-04-29"
draft: false
tags: ["questions"]
category: "questions"
---


Compare Adam and SGD through controlled optimization and generalization experiments rather than assuming one optimizer universally wins.

## Problem

**Context.** A workload appears to generalize differently under two optimizers; the coordinate-change fixture below probes one proposed sharpness explanation.

**Task.** Design a fair comparison, separate observed performance from its explanation, and analyze whether reparameterization changes the predictor's outputs.
Identify evidence that would support or reject a proposed mechanism.

**Constraints.** Make data splits, initialization, tuning budgets, schedules, and regularization conventions explicit.
Parameter-space curvature depends on coordinates; do not treat it alone as proof of different function-space behavior or generalization.

A toy predictor outputs $w$, with loss $L(w)=(w-1)^2/2$.
Reparameterize it as $w=10\phi$.
Compare the equivalent minima $w=1$ and $\phi=0.1$.
Does a sharper parameter-space curve imply different predictions?

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>
<span id="what-an-l6-answer-sounds-like"></span>

<!-- visual:optimizer-sharpness-reparameterization -->
<figure class="learning-figure plot-panel" aria-labelledby="optimizer-sharpness-visual-title">
	<p class="visual-kicker">The sharpness caveat</p>
	<p class="visual-title" id="optimizer-sharpness-visual-title">The basin gets sharper without changing the model's predictions</p><div class="visual-argument" data-visual-argument="optimizer-sharpness-reparameterization" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 439" role="img" aria-label="A stretched coordinate makes the same basin look sharper. These two parameter coordinates represent the same scalar prediction error. Curvature alone is not invariant to reparameterization."><title>A stretched coordinate makes the same basin look sharper</title><desc>These two parameter coordinates represent the same scalar prediction error. Curvature alone is not invariant to reparameterization.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">-2</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">2</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">4</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="0.5" d="M224.25 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><text x="224.25" y="245" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="224.25" dy="0">0.5</tspan></text><path data-proof-series="L(a) = a squared" d="M48 50L52.40625 60.458984375L56.8125 70.5859375L61.21875 80.380859375L65.625 89.84375L70.03125 98.974609375L74.4375 107.7734375L78.84375 116.240234375L83.25 124.375L87.65625 132.177734375L92.0625 139.6484375L96.46875 146.787109375L100.875 153.59375L105.28125 160.068359375L109.6875 166.2109375L114.09375 172.021484375L118.5 177.5L122.90625 182.646484375L127.3125 187.4609375L131.71875 191.943359375L136.125 196.09375L140.53125 199.912109375L144.9375 203.3984375L149.34375 206.552734375L153.75 209.375L158.15625 211.865234375L162.5625 214.0234375L166.96875 215.849609375L171.375 217.34375L175.78125 218.505859375L180.1875 219.3359375L184.59375 219.833984375L189 220L193.40625 219.833984375L197.8125 219.3359375L202.21875 218.505859375L206.625 217.34375L211.03125 215.849609375L215.4375 214.0234375L219.84375 211.865234375L224.25 209.375L228.65625 206.552734375L233.0625 203.3984375L237.46875 199.912109375L241.875 196.09375L246.28125 191.943359375L250.6875 187.4609375L255.09375 182.646484375L259.5 177.5L263.90625 172.021484375L268.3125 166.2109375L272.71875 160.068359375L277.125 153.59375L281.53125 146.787109375L285.9375 139.6484375L290.34375 132.177734375L294.75 124.375L299.15625 116.240234375L303.5625 107.7734375L307.96875 98.974609375L312.375 89.84375L316.78125 80.380859375L321.1875 70.5859375L325.59375 60.458984375L330 50" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="0.5" data-proof-y="0.25"><circle cx="224.25" cy="209.375" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="L(b) = (2b) squared; a = 2b" d="M118.5 50L120.703125 60.458984375L122.90625 70.5859375L125.109375 80.380859375L127.3125 89.84375L129.515625 98.974609375L131.71875 107.7734375L133.921875 116.240234375L136.125 124.375L138.328125 132.177734375L140.53125 139.6484375L142.734375 146.787109375L144.9375 153.59375L147.140625 160.068359375L149.34375 166.2109375L151.546875 172.021484375L153.75 177.5L155.953125 182.646484375L158.15625 187.4609375L160.359375 191.943359375L162.5625 196.09375L164.765625 199.912109375L166.96875 203.3984375L169.171875 206.552734375L171.375 209.375L173.578125 211.865234375L175.78125 214.0234375L177.984375 215.849609375L180.1875 217.34375L182.390625 218.505859375L184.59375 219.3359375L186.796875 219.833984375L189 220L191.203125 219.833984375L193.40625 219.3359375L195.609375 218.505859375L197.8125 217.34375L200.015625 215.849609375L202.21875 214.0234375L204.421875 211.865234375L206.625 209.375L208.828125 206.552734375L211.03125 203.3984375L213.234375 199.912109375L215.4375 196.09375L217.640625 191.943359375L219.84375 187.4609375L222.046875 182.646484375L224.25 177.5L226.453125 172.021484375L228.65625 166.2109375L230.859375 160.068359375L233.0625 153.59375L235.265625 146.787109375L237.46875 139.6484375L239.671875 132.177734375L241.875 124.375L244.078125 116.240234375L246.28125 107.7734375L248.484375 98.974609375L250.6875 89.84375L252.890625 80.380859375L255.09375 70.5859375L257.296875 60.458984375L259.5 50" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="0.5" data-proof-y="1"><circle cx="224.25" cy="177.5" r="5" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">parameter coordinate</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">loss</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">L(a) = a squared</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">0.25 at 0.5</tspan></text><path d="M20 368L48 368" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="372" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">L(b) = (2b) squared; a = 2b</tspan></text><text x="60" y="395" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="60" dy="0">1 at 0.5</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-scroll">
		<svg viewBox="0 0 360 326" role="img" aria-labelledby="optimizer-sharpness-svg-title optimizer-sharpness-svg-desc">
			<title id="optimizer-sharpness-svg-title">Exact quadratic profiles under w equals ten times phi</title>
			<desc id="optimizer-sharpness-svg-desc">Both plots use the same numerical horizontal perturbation scale and the same loss scale. The upper profile is one half delta w squared, with curvature one. The lower is fifty delta phi squared, with curvature one hundred, clipped to its plot. Both minima represent the same constant predictor one and have loss zero. Changing parameter coordinates changed curvature without changing predictions.</desc>
			<defs><clipPath id="optimizer-phi-plot"><rect x="42" y="216" width="290" height="72"></rect></clipPath></defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="124" rx="3"></rect>
			<text class="viz-axis-label" x="28" y="32">PARAMETERIZATION A · BROAD PROFILE</text>
			<path class="viz-axis" d="M42 44V116H332"></path>
			<path class="viz-roc-curve" d="M60 60Q180 160 300 60"></path>
			<path class="viz-baseline" d="M42 110H332"></path>
			<circle class="viz-operating-point" cx="180" cy="110" r="4"></circle>
			<text class="viz-label" x="49" y="57">loss</text>
			<text class="viz-label" x="327" y="130" text-anchor="end">Δw: -0.5 to +0.5</text>
			<text class="viz-callout" x="190" y="103">curvature 1</text>
			<path class="viz-axis" d="M180 144V174"></path>
			<path class="viz-axis" d="M174 168L180 174L186 168"></path>
			<text class="viz-callout" x="190" y="153">w = 10φ</text>
			<text class="viz-label" x="190" y="169">same function and predictions</text>
			<rect class="viz-plot-bg" x="12" y="184" width="336" height="124" rx="3"></rect>
			<text class="viz-axis-label" x="28" y="204">PARAMETERIZATION B · NARROW PROFILE</text>
			<path class="viz-axis" d="M42 216V288H332"></path>
			<path class="viz-roc-curve" d="M60 -4718Q180 5282 300 -4718" clip-path="url(#optimizer-phi-plot)"></path>
			<path class="viz-baseline" d="M42 282H332"></path>
			<circle class="viz-operating-point" cx="180" cy="282" r="4"></circle>
			<text class="viz-label" x="49" y="229">loss</text>
			<text class="viz-label" x="327" y="302" text-anchor="end">Δφ: -0.5 to +0.5</text>
			<text class="viz-callout" x="206" y="275">curvature 100</text>
			<text class="viz-callout" x="180" y="322" text-anchor="middle">apparent sharpness changed; model behavior did not</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> coordinate curvature changes from one to one hundred. Both minima still predict one. These exact toy curves show coordinate dependence, not measured optimizer performance.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="optimizer-sharpness-reparameterization"><strong>Check:</strong> Did predictions change when the plotted basin became sharper?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="optimizer-sharpness-reparameterization">These two parameter coordinates represent the same scalar prediction error. Curvature alone is not invariant to reparameterization.</span></figcaption>
</figure>
<p class="diagram-source">Reparameterization caveat: <a href="https://arxiv.org/abs/1703.04933">Dinh et al. (2017)</a>. This scalar construction is calculated independently.</p>

## Calculate the curvature

In $w$ coordinates, $L''(w)=1$.
In $\phi$ coordinates, $L(\phi)=(10\phi-1)^2/2$, so $L''(\phi)=100$.
The same predictor has the same performance on any fixed evaluation set.
Raw coordinate sharpness alone therefore cannot establish worse generalization.
This does not prove every normalized sharpness measure is useless.

## Establish the comparison

Record data splits, architecture, augmentations, initialization, batch, precision, schedules, decay and the resource budget.
Give both optimizer families a declared tuning opportunity.
Select on validation evidence, then compare on independent evaluation with a planned repetition and uncertainty rule.

| Claim | Evidence needed | Alternative explanation | Discriminating check |
| --- | --- | --- | --- |
| One optimizer generalizes better | Held-out results under the comparison contract | Unequal tuning or augmentation | Match opportunity and repeat |
| The gap is caused by sharpness | A defined measure with coordinate controls | Reparameterization changes curvature | Test equivalent predictors |
| AdamW fixes the entire gap | Workload-specific tuned comparisons | Schedule or decay changed too | Separate those interventions |

## Change the comparison

### 1. Limit elapsed time

**New task:** choose a procedure under a one-hour training budget, not equal steps.

**Picture and change:** compare quality at the deadline and record steps, tokens and overhead.
Match tuning opportunity as well as final-run time.

**Check:** hypothetical step times of $0.25$ and $0.5$ seconds permit $14400$ versus $7200$ steps, ignoring overhead.
Quality after $10000$ steps alone cannot decide the one-hour question.

**Use when:** time or compute limits the real decision.

### 2. Separate adaptive decay from independent shrinkage

**New task:** determine whether a comparison changes regularization as well as optimizer family.

**Picture and change:** a coupled penalty adds $\lambda\theta$ before adaptive scaling.
Decoupled decay shrinks parameters independently of that scaling.
Even holding a diagonal step scale $D$ fixed, the updates differ:

$$
	heta_{\mathrm{coupled}}=\theta-\eta D(g+\lambda\theta),\qquad
	heta_{\mathrm{decoupled}}=(1-\eta\lambda)\theta-\eta Dg.
$$

**Check:** with $\theta=(1,1)$, $g=0$, $D=\operatorname{diag}(1,0.1)$ and $\eta=\lambda=0.1$, results are $(0.99,0.999)$ versus $(0.99,0.99)$.
In Adam, the coupled term also enters the accumulated moments.

**Use when:** isolating decay effects. Retune appropriate strengths before attributing the gap to moment normalization.

## Unequal search can create the gap

One family receiving a large rate sweep while another receives one default does not isolate optimizer quality.
AdamW changes the update, but does not guarantee better generalization everywhere.

## Try interpreting equal training loss

Does equal training loss plus a sharper-looking minimum establish worse generalization?

<details>
<summary>Check your answer</summary>

No. Measure independent predictive performance and define the curvature measure and its coordinate dependence.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Workload and batch regime

Vision, language and large-batch workloads require specific comparison evidence.
Neither optimizer family wins by definition, and Adam is not identical to AdamW.
[Decoupled weight decay](https://arxiv.org/abs/1711.05101) changes adaptive regularization; its benchmark results are not universal rankings.
[LAMB](https://arxiv.org/abs/1904.00962) adds layer-wise adaptation for large-batch training under a particular recipe.
Its existence does not show that ordinary Adam or SGD always fails above a fixed batch size.
Gradient-noise and implicit-bias explanations need defined assumptions and tests, not just a valley illustration.

## Common follow-up

For an AdamW schedule, state the intended horizon, warmup, peak, final rate and selection procedure.
Warmup-plus-cosine and stable-plus-final-decay are alternatives to test under matched budgets.
Fixed warmup fractions and rate multipliers are not universal guarantees.

Related: [Adam and AdamW](/concepts/adam-and-adamw/), [learning-rate selection](/questions/how-to-choose-learning-rate/) and [regularization](/concepts/regularization/).

</div>
