---
title: "Cross-entropy and softmax"
description: "The pairing isn't arbitrary. Cross-entropy is the negative log-likelihood under a categorical distribution, and the softmax+CE gradient simplifies to (p − y), which is why it's stable."
date: "2025-09-05"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

**Cross-entropy loss** is the negative log-likelihood of the true class under a probability distribution predicted by the model. **Softmax** is the standard parameterization that turns logits into a categorical distribution. The two are nearly always paired because the math composes cleanly.

Almost every classification model uses softmax + cross-entropy. The reasons are not arbitrary:

1. **Cross-entropy is the right loss for classification under MLE**. If you assume your label is a sample from a categorical distribution and you want maximum likelihood, the loss is exactly cross-entropy.
2. **The gradient simplifies to (p − y)**. The composition of softmax and cross-entropy has the unique property that the gradient of the loss with respect to the logits is `softmax_output − one_hot_label`. Numerically stable, easy to compute.
3. **MSE on classification has vanishing gradients for confident-but-wrong predictions**. Cross-entropy doesn't; the gradient stays large precisely when the model is most wrong.

## The math, briefly

For C classes and a single example with true class y:

**Softmax**: `p_i = exp(z_i) / sum_j exp(z_j)` for logits z = (z_1, ..., z_C).

**Cross-entropy**: `L = -log p_y = -z_y + log(sum_j exp(z_j))`.

Note the second form (log-sum-exp): this is the numerically stable way to compute cross-entropy, you never explicitly form the softmax, you compute LSE on the logits directly. PyTorch's `nn.CrossEntropyLoss` does this.

**Gradient w.r.t. logits**: `dL/dz_i = p_i − y_i` where y is one-hot. Three lines of algebra; the cleanest gradient in deep learning.

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
	<figcaption><strong>Read it this way:</strong> move left toward a confidently wrong prediction. For cross-entropy, the true-logit gradient <code>q − 1</code> approaches −1, so the correction stays strong. For half squared error on this binary probability, the chain rule adds the saturating factor <code>q(1 − q)</code>; its gradient magnitude <code>q(1 − q)²</code> collapses toward zero. The dashed curve is the binary slice used to expose that extra factor; multiclass softmax plus cross-entropy gives <code>dL/dzᵢ = pᵢ − yᵢ</code>.</figcaption>
</figure>

## What an interviewer expects you to say

If asked "why softmax + cross-entropy":

1. State the MLE interpretation.
2. State the gradient simplification.
3. Mention the numerical stability of computing them jointly (log-sum-exp trick).
4. Mention the contrast with MSE on classification (vanishing gradients).

Bonus depth: temperature `tau` (dividing logits by `tau` before softmax) controls the sharpness of the distribution, high `tau` makes it more uniform, low `tau` makes it more peaked. Used in distillation (high `tau` to extract more information from the teacher) and in sampling from LLMs (low `tau` for greedy-like behavior).

## Common confusions

- **"Softmax" vs "softmax + cross-entropy" as separate operations.** Conceptually distinct, but in practice always computed jointly because the joint gradient is so much cleaner.
- **Computing cross-entropy on probabilities you already softmaxed.** PyTorch's `nn.CrossEntropyLoss` takes *logits*, not probabilities. Passing `softmax(logits)` will give wrong gradients and (often silently) bad training.
- **"Cross-entropy" vs "categorical cross-entropy" vs "binary cross-entropy".** All the same idea; "binary" is just C=2 (often parameterized with sigmoid instead of softmax).
- **MSE for classification "as a baseline".** Don't. Vanishing gradients for confident-wrong predictions; train slowly.

## Numerical stability: the LSE trick

The naive `log(sum_j exp(z_j))` overflows for large z_j. The fix:

```
log_sum_exp(z) = max(z) + log(sum_j exp(z_j - max(z)))
```

Subtract the max before exponentiating; add back outside the log. The maximum exponential argument is now 0; no overflow.

Every framework's `cross_entropy_loss` does this internally. If you write your own, you must too.

## Why MSE fails on classification

MSE gradient for a single output is proportional to `(p - y) * p * (1-p)` (when paired with sigmoid). When the model is *very* confident and wrong (`p ≈ 1` for the wrong class), the `p*(1-p)` term vanishes, the model can't learn its way out of the bad prediction. Cross-entropy's gradient is `(p - y)` directly, which stays large.

This is why no one uses MSE for classification despite the textbook deriving it for completeness.

---

*Related: [BatchNorm vs LayerNorm](/concepts/batchnorm-vs-layernorm/). Related interview: [Why does dropout work?](/questions/why-does-dropout-work/).*
