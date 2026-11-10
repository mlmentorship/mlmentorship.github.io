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
	<p class="visual-title" id="optimizer-sharpness-visual-title">The basin gets sharper without changing the model's predictions</p><div class="visual-argument" data-visual-argument="optimizer-sharpness-reparameterization" data-argument-kind="curve" data-argument-composition="coordinate-relabel"><svg viewBox="0 0 360 960" role="img" aria-label="A stretched coordinate makes the same basin look sharper. Both plots use the same loss and coordinate scales. A smaller coordinate change now covers the same prediction change. The second basin looks steeper, yet matching predictions keep the same loss. Curvature depends on the coordinates, so sharpness alone cannot establish which optimizer generalizes better."><title>A stretched coordinate makes the same basin look sharper</title><desc>Both plots use the same loss and coordinate scales. A smaller coordinate change now covers the same prediction change. The second basin looks steeper, yet matching predictions keep the same loss. Curvature depends on the coordinates, so sharpness alone cannot establish which optimizer generalizes better.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">The same prediction has different</tspan><tspan x="20" dy="18">coordinate labels</tspan></text><g data-picture-sharpness-scale="1" data-picture-coordinate-curvature="2" data-picture-sharpness-prediction="2" data-picture-sharpness-coordinate="2" data-picture-sharpness-loss="4"><text x="20" y="104" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Coordinate a: prediction equals a</tspan></text><text x="20" y="141" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">loss</tspan></text><path d="M45 340L315 340" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="345" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">0</tspan></text><path d="M45 295L315 295" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="300" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">4</tspan></text><path d="M45 250L315 250" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="255" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">8</tspan></text><path d="M45 205L315 205" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="210" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">12</tspan></text><path d="M45 160L315 160" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="165" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">16</tspan></text><path d="M45 160L45 340" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M45 340L47.7 339.982L50.4 339.928L53.1 339.838L55.8 339.712L58.5 339.55L61.2 339.352L63.900000000000006 339.118L66.6 338.848L69.3 338.542L72 338.2L74.7 337.822L77.4 337.408L80.1 336.958L82.80000000000001 336.472L85.5 335.95L88.2 335.392L90.9 334.798L93.6 334.168L96.3 333.502L99 332.8L101.69999999999999 332.062L104.4 331.288L107.1 330.478L109.8 329.632L112.5 328.75L115.2 327.832L117.9 326.878L120.60000000000001 325.888L123.3 324.862L126 323.8L128.7 322.702L131.4 321.568L134.10000000000002 320.398L136.8 319.192L139.5 317.95L142.2 316.672L144.9 315.358L147.6 314.008L150.3 312.622L153 311.2L155.7 309.742L158.39999999999998 308.248L161.1 306.718L163.8 305.152L166.5 303.55L169.2 301.912L171.89999999999998 300.238L174.6 298.528L177.3 296.782L180 295L182.7 293.182L185.4 291.328L188.1 289.438L190.8 287.512L193.5 285.55L196.20000000000002 283.552L198.89999999999998 281.51800000000003L201.6 279.448L204.29999999999998 277.342L207 275.2L209.7 273.022L212.4 270.808L215.1 268.558L217.8 266.272L220.5 263.95L223.20000000000002 261.592L225.9 259.198L228.60000000000002 256.768L231.29999999999998 254.30200000000002L234 251.8L236.7 249.262L239.4 246.688L242.1 244.07800000000003L244.8 241.43200000000002L247.5 238.75L250.2 236.03199999999998L252.9 233.27800000000002L255.6 230.488L258.3 227.66199999999998L261 224.79999999999998L263.70000000000005 221.902L266.4 218.96800000000002L269.1 215.998L271.79999999999995 212.99200000000002L274.5 209.95000000000002L277.2 206.872L279.9 203.758L282.6 200.608L285.3 197.422L288 194.2L290.70000000000005 190.94199999999998L293.4 187.648L296.1 184.31799999999998L298.79999999999995 180.952L301.5 177.55L304.2 174.112L306.9 170.638L309.6 167.12800000000001L312.3 163.582L315 160" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></path><circle cx="180" cy="295" r="7" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="45" y="370" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="45" dy="0">0</tspan></text><text x="112.5" y="370" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="112.5" dy="0">1</tspan></text><text x="180" y="370" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="180" dy="0">2</tspan></text><text x="247.5" y="370" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="247.5" dy="0">3</tspan></text><text x="315" y="370" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="315" dy="0">4</tspan></text><text x="20" y="399" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">coordinate a</tspan></text><text x="20" y="431" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Selected prediction 2; loss 4</tspan></text></g><g data-picture-sharpness-scale="2" data-picture-coordinate-curvature="8" data-picture-sharpness-prediction="2" data-picture-sharpness-coordinate="1" data-picture-sharpness-loss="4"><text x="20" y="454" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Coordinate b: prediction is twice b</tspan></text><text x="20" y="491" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">loss</tspan></text><path d="M45 690L315 690" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="695" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">0</tspan></text><path d="M45 645L315 645" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="650" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">4</tspan></text><path d="M45 600L315 600" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="605" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">8</tspan></text><path d="M45 555L315 555" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="560" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">12</tspan></text><path d="M45 510L315 510" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="37" y="515" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="37" dy="0">16</tspan></text><path d="M45 510L45 690" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M45 690L46.35 689.982L47.7 689.928L49.05 689.838L50.4 689.712L51.75 689.55L53.1 689.352L54.45 689.118L55.8 688.848L57.15 688.542L58.5 688.2L59.85 687.822L61.2 687.408L62.55 686.958L63.900000000000006 686.472L65.25 685.95L66.6 685.392L67.95 684.798L69.3 684.168L70.65 683.502L72 682.8L73.35 682.062L74.7 681.288L76.05 680.478L77.4 679.632L78.75 678.75L80.1 677.832L81.45 676.878L82.80000000000001 675.888L84.15 674.862L85.5 673.8L86.85 672.702L88.2 671.568L89.55000000000001 670.398L90.9 669.192L92.25 667.95L93.6 666.672L94.95 665.358L96.3 664.008L97.65 662.622L99 661.2L100.35 659.742L101.69999999999999 658.248L103.05 656.718L104.4 655.152L105.75 653.55L107.1 651.912L108.44999999999999 650.238L109.8 648.528L111.15 646.782L112.5 645L113.85 643.182L115.2 641.328L116.55 639.438L117.9 637.512L119.25 635.55L120.60000000000001 633.552L121.94999999999999 631.518L123.3 629.448L124.64999999999999 627.342L126 625.2L127.35 623.022L128.7 620.808L130.05 618.558L131.4 616.2719999999999L132.75 613.95L134.10000000000002 611.592L135.45 609.198L136.8 606.768L138.14999999999998 604.302L139.5 601.8L140.85 599.262L142.2 596.688L143.55 594.078L144.9 591.432L146.25 588.75L147.6 586.032L148.95 583.278L150.3 580.4879999999999L151.65 577.662L153 574.8L154.35000000000002 571.9019999999999L155.7 568.9680000000001L157.05 565.998L158.39999999999998 562.992L159.75 559.95L161.1 556.8720000000001L162.45 553.758L163.8 550.608L165.15 547.422L166.5 544.2L167.85000000000002 540.942L169.2 537.648L170.55 534.318L171.89999999999998 530.952L173.25 527.55L174.6 524.112L175.95 520.638L177.3 517.128L178.65 513.582L180 510" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5"></path><circle cx="112.5" cy="645" r="7" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="45" y="720" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="45" dy="0">0</tspan></text><text x="112.5" y="720" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="112.5" dy="0">1</tspan></text><text x="180" y="720" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="180" dy="0">2</tspan></text><text x="247.5" y="720" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="247.5" dy="0">3</tspan></text><text x="315" y="720" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="315" dy="0">4</tspan></text><text x="20" y="749" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">coordinate b</tspan></text><text x="20" y="781" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Selected prediction 2; loss 4</tspan></text></g><text x="20" y="845" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same prediction; same loss of four.</tspan></text><text x="20" y="906" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Coordinate curvatures differ: two</tspan><tspan x="20" dy="18">versus eight.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> coordinate curvature changes from one to one hundred. Both minima still predict one. These exact toy curves show coordinate dependence, not measured optimizer performance.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="optimizer-sharpness-reparameterization"><strong>Check:</strong> Did predictions change when the plotted basin became sharper?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="optimizer-sharpness-reparameterization">Both plots use the same loss and coordinate scales. A smaller coordinate change now covers the same prediction change. The second basin looks steeper, yet matching predictions keep the same loss. Curvature depends on the coordinates, so sharpness alone cannot establish which optimizer generalizes better. <strong>If only the coordinate units change, do predictions change?</strong></span></figcaption>
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
