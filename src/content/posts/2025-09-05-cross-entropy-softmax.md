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
	<p class="visual-title" id="ce-gradient-visual-title">Why does cross-entropy keep correcting a confident wrong prediction?</p>
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
	<figcaption><strong>Read it this way:</strong> move left toward confident error. Cross-entropy's gradient approaches minus one. Probability-MSE includes the saturating sigmoid derivative, so its magnitude approaches zero. The dashed curve describes this binary comparison, not every possible squared-error model.</figcaption>
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
