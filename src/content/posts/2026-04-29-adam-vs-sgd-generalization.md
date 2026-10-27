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
	<p class="visual-title" id="optimizer-sharpness-visual-title">One predictor can look flat or sharp under different parameterizations.</p>
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
	<figcaption><strong>Read it this way:</strong> coordinate curvature changes from one to one hundred. Both minima still predict one. These exact toy curves show coordinate dependence, not measured optimizer performance.</figcaption>
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
