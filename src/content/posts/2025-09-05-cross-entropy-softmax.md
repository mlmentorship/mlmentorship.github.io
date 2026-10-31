---
title: "Cross-entropy and softmax"
description: "Derive categorical negative log-likelihood and its logit gradient, then change target smoothing and temperature without losing the normalization contract."
date: "2025-09-05"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Categorical cross-entropy is negative log-likelihood when one class label is observed from a predicted categorical distribution.
Softmax maps finite logits to that distribution; differentiating their composition gives a prediction-minus-target logit gradient.
Stable evaluation comes from shifted log-normalization, not from the gradient formula alone.

## The math, briefly

Let $z\in\mathbb R^C$ be one logit vector and $y$ its observed class index.
Define $p_i=e^{z_i}/\sum_j e^{z_j}$.
Then

$$
L=-\log p_y=-z_y+\log\sum_j e^{z_j},\qquad
\frac{\partial L}{\partial z_i}=p_i-\mathbf1[i=y].
$$

The derivative of the log-normalizer is $e^{z_i}/\sum_j e^{z_j}=p_i$.
The observed-class term contributes $-1$ only at index $y$.
For logits $(\log3,0)$ and class zero, $p=(3/4,1/4)$, loss is $\log(4/3)$, and gradient is $(-1/4,1/4)$.

The diagram instead uses a binary true-class probability $q=0.01$.
Cross-entropy's true-logit gradient is $q-1=-0.99$.
Half squared error on the sigmoid probability introduces another derivative factor, giving magnitude $q(1-q)^2=0.009801$.

<!-- visual:cross-entropy-confident-wrong-gradient -->
<figure class="learning-figure plot-panel" aria-labelledby="ce-gradient-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ce-gradient-visual-title">Cross-entropy keeps a strong gradient for a confident wrong answer</p><div class="visual-argument" data-visual-argument="cross-entropy-confident-wrong-gradient" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 439" role="img" aria-label="Confidence in the wrong answer need not erase the gradient. For a positive label, binary cross-entropy has logit gradient p - 1. The squared-error gradient also contains the small sigmoid derivative."><title>Confidence in the wrong answer need not erase the gradient</title><desc>For a positive label, binary cross-entropy has logit gradient p - 1. The squared-error gradient also contains the small sigmoid derivative.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">0</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">1</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">1</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="0.05" d="M62.1 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><path data-proof-series="Cross-entropy: 1 - p" d="M48 50L52.40625 52.65625L56.8125 55.3125L61.21875 57.96875L62.1 58.5L65.625 60.625L70.03125 63.28125L74.4375 65.9375L78.84375 68.59375L83.25 71.25L87.65625 73.90625L92.0625 76.5625L96.46875 79.21875L100.875 81.875L105.28125 84.53125L109.6875 87.1875L114.09375 89.84375L118.5 92.5L122.90625 95.15625L127.3125 97.8125L131.71875 100.46875L136.125 103.125L140.53125 105.78125L144.9375 108.4375L149.34375 111.09375L153.75 113.75L158.15625 116.40625L162.5625 119.0625L166.96875 121.71875L171.375 124.375L175.78125 127.03125L180.1875 129.6875L184.59375 132.34375L189 135L193.40625 137.65625L197.8125 140.3125L202.21875 142.96875L206.625 145.625L211.03125 148.28125L215.4375 150.9375L219.84375 153.59375L224.25 156.25L228.65625 158.90625L233.0625 161.5625L237.46875 164.21875L241.875 166.875L246.28125 169.53125L250.6875 172.1875L255.09375 174.84375L259.5 177.5L263.90625 180.15625L268.3125 182.8125L272.71875 185.46875L277.125 188.125L281.53125 190.78125L285.9375 193.4375L290.34375 196.09375L294.75 198.75L299.15625 201.40625L303.5625 204.0625L307.96875 206.71875L312.375 209.375L316.78125 212.03125L321.1875 214.6875L325.59375 217.34375L330 220" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="0.05" data-proof-y="0.95"><circle cx="62.1" cy="58.5" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Squared error: 2p(1-p)^2" d="M48 220L52.40625 214.8522186279297L56.8125 210.0286865234375L61.21875 205.52162170410156L62.1 204.6575L65.625 201.3232421875L70.03125 197.42576599121094L74.4375 193.8214111328125L78.84375 190.5023956298828L83.25 187.4609375L87.65625 184.6892547607422L92.0625 182.1795654296875L96.46875 179.92408752441406L100.875 177.9150390625L105.28125 176.14463806152344L109.6875 174.6051025390625L114.09375 173.2886505126953L118.5 172.1875L122.90625 171.2938690185547L127.3125 170.5999755859375L131.71875 170.09803771972656L136.125 169.7802734375L140.53125 169.63890075683594L144.9375 169.6661376953125L149.34375 169.8542022705078L153.75 170.1953125L158.15625 170.6816864013672L162.5625 171.3055419921875L166.96875 172.05909729003906L171.375 172.9345703125L175.78125 173.92417907714844L180.1875 175.0201416015625L184.59375 176.2146759033203L189 177.5L193.40625 178.8683319091797L197.8125 180.3118896484375L202.21875 181.82289123535156L206.625 183.3935546875L211.03125 185.01609802246094L215.4375 186.6827392578125L219.84375 188.3856964111328L224.25 190.1171875L228.65625 191.8694305419922L233.0625 193.6346435546875L237.46875 195.40504455566406L241.875 197.1728515625L246.28125 198.93028259277344L250.6875 200.6695556640625L255.09375 202.3828887939453L259.5 204.0625L263.90625 205.7006072998047L268.3125 207.2894287109375L272.71875 208.82118225097656L277.125 210.2880859375L281.53125 211.68235778808594L285.9375 212.9962158203125L290.34375 214.2218780517578L294.75 215.3515625L299.15625 216.3774871826172L303.5625 217.2918701171875L307.96875 218.08692932128906L312.375 218.7548828125L316.78125 219.28794860839844L321.1875 219.6783447265625L325.59375 219.9182891845703L330 220" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="0.05" data-proof-y="0.09025"><circle cx="62.1" cy="204.6575" r="5" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">predicted positive probability</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">absolute logit gradient</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Cross-entropy: 1 - p</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">0.95 at 0.05</tspan></text><path d="M20 368L48 368" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="372" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Squared error: 2p(1-p)^2</tspan></text><text x="60" y="395" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="60" dy="0">0.09025 at 0.05</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 300" role="img" aria-labelledby="ce-gradient-svg-title ce-gradient-svg-desc">
		<title id="ce-gradient-svg-title">Cross-entropy and squared-error logit gradients as true-class probability changes</title>
		<desc id="ce-gradient-svg-desc">An original plot for a binary output with true label one. The horizontal axis is true-class probability q from zero to one, and the vertical axis is absolute gradient with respect to its logit. Cross-entropy is a solid straight line with magnitude one minus q, so at q equals 0.01 its gradient is 0.99. Half squared error on the probability is a dashed low curve with magnitude q times one minus q squared, so at q equals 0.01 its gradient is about 0.0098. The extra sigmoid derivative makes squared error nearly stop when the prediction is confidently wrong.</desc>
		<rect class="viz-plot-bg" x="48" y="25" width="297" height="215" rx="4"></rect>
		<path class="viz-gridline" d="M60 35H335 M60 130H335 M60 225H335 M60 35V225 M128.75 35V225 M197.5 35V225 M266.25 35V225 M335 35V225"></path>
		<path class="viz-axis" d="M60 35V225H335"></path>
		<path class="viz-roc-curve" d="M60 35L335 225"></path>
		<path class="viz-pr-curve" style="stroke-dasharray:7 5" d="M60 225 L87.5 209.6 L115 200.7 L142.5 197.1 L170 197.6 L197.5 201.3 L225 206.8 L252.5 213 L280 218.9 L307.5 223.3 L335 225"></path>
		<path class="viz-operating-guide" d="M62.75 35V225"></path>
		<circle class="viz-operating-point" cx="62.75" cy="36.9" r="4"></circle>
		<circle class="viz-operating-point" cx="62.75" cy="223.1" r="4"></circle>
		<text class="viz-callout" x="92" y="53">CE: |g| = 0.99</text>
		<text class="viz-callout" x="90" y="216">MSE: |g| ≈ 0.0098</text>
		<text class="viz-callout" x="150" y="91">cross-entropy |g| = 1 − q (solid)</text>
		<text class="viz-callout" x="90" y="172">probability-MSE (dashed)</text>
		<text class="viz-callout" x="90" y="187">|g| = q(1 − q)²</text>
		<text class="viz-label" x="52" y="39" text-anchor="end">1.0</text>
		<text class="viz-label" x="52" y="134" text-anchor="end">0.5</text>
		<text class="viz-label" x="52" y="229" text-anchor="end">0</text>
		<text class="viz-label" x="60" y="254" text-anchor="middle">0</text>
		<text class="viz-label" x="128.75" y="254" text-anchor="middle">0.25</text>
		<text class="viz-label" x="197.5" y="254" text-anchor="middle">0.50</text>
		<text class="viz-label" x="266.25" y="254" text-anchor="middle">0.75</text>
		<text class="viz-label" x="335" y="254" text-anchor="middle">1.0</text>
		<text class="viz-axis-label" x="60" y="17">logit-gradient magnitude |g|</text>
		<text class="viz-axis-label" x="197.5" y="280" text-anchor="middle">true-class probability q · true label y = 1</text>
	</svg>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> move left toward confident error. Cross-entropy's gradient approaches minus one. Probability-MSE includes the saturating sigmoid derivative, so its magnitude approaches zero. The dashed curve describes this binary comparison, not every possible squared-error model.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="cross-entropy-confident-wrong-gradient"><strong>Check:</strong> Why is the cross-entropy logit gradient large where the probability-squared-error gradient nearly vanishes?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="cross-entropy-confident-wrong-gradient">For a positive label, binary cross-entropy has logit gradient p - 1. The squared-error gradient also contains the small sigmoid derivative.</span></figcaption>
</figure>

## What an interviewer expects you to say

State the observation model, derive the log-normalizer gradient, declare the reduction, and explain stable evaluation.
For independent examples, summed negative log-likelihood is the joint training objective; a mean divides its gradients by sample count.
For multilabel tasks with independent binary targets, categorical normalization changes the model and is generally inappropriate.

## Change the Requirement

### Smooth the Target

**New task:** Replace a one-hot target with $t=(1-\varepsilon)e_y+\varepsilon\mathbf1/C$, where $0\le\varepsilon\le1$.

**Picture and change:** Optimize $-\sum_i t_i\log p_i$; normalization $\sum_i t_i=1$ preserves gradient $p-t$.

**Check:** With $p=(0.75,0.25)$, class zero, and $\varepsilon=0.1$, target $(0.95,0.05)$ gives gradient $(-0.20,0.20)$.

**Use when:** A deliberately softened training target is appropriate.
The target and its interpretation have changed; improved calibration is not guaranteed by the algebra.

### Change Temperature

**New task:** Use probabilities $p_T=\operatorname{softmax}(z/T)$, with fixed $T>0$.

**Picture and change:** The chain rule adds $1/T$: $\nabla_zL=(p_T-e_y)/T$.

**Check:** For $z=(\log3,0)$ and $T=2$, $p_{T,0}=\sqrt3/(1+\sqrt3)\approx0.6340$.
The gradient is approximately $(-0.1830,0.1830)$.

**Use when:** The task specifies temperature-scaled training or inference probabilities.
Distillation may additionally scale its loss; state that convention separately.

## Common confusions

- **Wrong API input.** A logits-based cross-entropy API does not expect already normalized probabilities.
- **Wrong task.** Multiclass normalization and independent multilabel Bernoulli losses encode different outcome spaces.
- **Unnormalized targets.** For general target weights $t$, the gradient is $(\sum_i t_i)p-t$, not automatically $p-t$.

## Numerical stability: the LSE trick

Let $m=\max_i z_i$ and evaluate the shifted expression:

```
log_sum_exp(z) = max(z) + log(sum_j exp(z_j - max(z)))
```

For the loss itself, use $(m-z_y)+\log\sum_i e^{z_i-m}$.
This avoids exponentiating positive shifted logits and avoids subtracting two large log-normalized quantities afterward.
For logits $(1000,1000)$, class-zero loss is $\log2$, despite naive exponentiation overflowing.
Nonfinite logits, unrepresentable differences, and all-masked rows still need a declared policy.

<!-- learning:core-end -->
<div data-learning-extension>

## Why MSE fails on classification

The heading describes sigmoid saturation in the displayed example, not a universal impossibility theorem.
Squared probability error is related to the Brier score and can be a meaningful evaluation or training objective.
Its optimization behavior differs because the sigmoid derivative can suppress a confident-error logit update.
Cross-entropy avoids that extra factor for logits, but parameter gradients can still vanish through other network operations.

## Reconstruct

<details>
<summary>Does the simple logit gradient guarantee numerical stability or successful training?</summary>

No. Naive exponentiation can overflow before a gradient is computed.
Stable normalization addresses that arithmetic issue; optimization, model suitability, and upstream derivatives remain separate concerns.

</details>

---

*Related: [BatchNorm vs LayerNorm](/concepts/batchnorm-vs-layernorm/). Related interview: [Why does dropout work?](/questions/why-does-dropout-work/).*

</div>
