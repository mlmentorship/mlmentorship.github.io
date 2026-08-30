---
title: "Calibration: when your model says 80% it should be right 80% of the time"
description: "Accuracy isn't enough; you also want predictions to mean what they say. Calibration is the difference."
date: "2026-03-01"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

A model is well-calibrated if among predictions made with confidence p, the fraction that are correct is also p. A model can be accurate but poorly calibrated, or calibrated but inaccurate; both matter for production.

Many production systems consume model probabilities, not just classifications. Examples:
- Threshold tuning for downstream actions (flag for review if probability &gt; 0.9).
- Combining multiple models (you need probabilities to be on the same scale).
- Decision-making under uncertainty (expected value calculations require true probabilities).
- User-facing confidence displays.

Uncalibrated scores cause downstream failures. A 90% confidence prediction right 60% of the time produces wrong decisions.

## Measuring calibration

The standard tool: **reliability diagram + expected calibration error (ECE)**.

1. Bin predictions by confidence (e.g., 10 bins: 0-0.1, 0.1-0.2, ..., 0.9-1.0).
2. For each bin, compute (a) the average predicted confidence and (b) the actual accuracy.
3. Reliability diagram: plot accuracy vs confidence. Perfect calibration is the diagonal.
4. ECE: weighted average of |accuracy − confidence| across bins.

Typical interpretation:
- **Underconfident**: accuracy &gt; confidence (model is more right than it claims).
- **Overconfident**: accuracy &lt; confidence (model is too sure of itself). Most modern deep nets are overconfident.

<!-- visual:calibration-reliability-gap -->
<figure class="learning-figure plot-panel" aria-labelledby="calibration-visual-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="calibration-visual-title">Calibration is the vertical gap between observed frequency and predicted confidence.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 310" role="img" aria-labelledby="calibration-svg-title calibration-svg-desc">
			<title id="calibration-svg-title">Reliability diagram with perfect calibration and an illustrative overconfident model</title>
			<desc id="calibration-svg-desc">The horizontal axis is mean predicted confidence and the vertical axis is observed fraction correct. A dashed diagonal from zero-zero to one-one represents perfect calibration. An original illustrative curve connects bin pairs 20 percent confidence and 12 percent correct, 40 and 28, 60 and 43, 80 and 60, and 90 and 72. Every point is below the diagonal, so confidence exceeds observed correctness and the model is overconfident. At 80 percent confidence, a vertical guide marks the 20 percentage-point gap down to 60 percent observed correctness. The region above the diagonal is labeled underconfident and the region below is labeled overconfident.</desc>
			<rect class="viz-plot-bg" x="54" y="28" width="272" height="222" rx="3"></rect>
			<path class="viz-gridline" d="M54 194.5H326 M54 139H326 M54 83.5H326 M122 28V250 M190 28V250 M258 28V250"></path>
			<path class="viz-axis" d="M54 28V250H326"></path>
			<path class="viz-baseline" d="M54 250L326 28"></path>
			<text class="viz-axis-label" x="232" y="62" transform="rotate(-39 232 62)">perfect: observed = predicted</text>
			<text class="viz-label" x="77" y="64">UNDERCONFIDENT</text>
			<text class="viz-label" x="246" y="220">OVERCONFIDENT</text>
			<path class="viz-pr-curve" d="M108.4 223.4 L162.8 187.8 L217.2 154.5 L271.6 116.8 L298.8 90.2"></path>
			<circle class="viz-operating-point" cx="108.4" cy="223.4" r="4"></circle>
			<circle class="viz-operating-point" cx="162.8" cy="187.8" r="4"></circle>
			<circle class="viz-operating-point" cx="217.2" cy="154.5" r="4"></circle>
			<circle class="viz-operating-point" cx="271.6" cy="116.8" r="5"></circle>
			<circle class="viz-operating-point" cx="298.8" cy="90.2" r="4"></circle>
			<path class="viz-operating-guide" d="M271.6 72.4V116.8"></path>
			<path class="viz-operating-guide" d="M266.6 72.4H276.6 M266.6 116.8H276.6"></path>
			<text class="viz-callout" x="263" y="92" text-anchor="end">20-point gap</text>
			<text class="viz-callout" x="263" y="107" text-anchor="end">80% vs 60%</text>
			<text class="viz-callout" x="152" y="211">illustrative bins</text>
			<text class="viz-label" x="50" y="268">0</text>
			<text class="viz-label" x="186" y="268">0.5</text>
			<text class="viz-label" x="320" y="268">1</text>
			<text class="viz-label" x="36" y="254">0</text>
			<text class="viz-label" x="30" y="143">0.5</text>
			<text class="viz-label" x="36" y="33">1</text>
			<text class="viz-axis-label" x="190" y="294" text-anchor="middle">mean predicted confidence</text>
			<text class="viz-axis-label" transform="translate(14 194) rotate(-90)">observed fraction correct</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> the dashed diagonal is perfect calibration. The illustrative bin pairs are (20%, 12%), (40%, 28%), (60%, 43%), (80%, 60%), and (90%, 72%). Because observed correctness is below confidence in every bin, this model is overconfident; points above the diagonal would be underconfident.</figcaption>
</figure>

## Why neural networks are overconfident

Modern neural networks (especially with high capacity and limited training data) tend to be highly overconfident. Common reasons:

- Cross-entropy loss minimization rewards confidence; the model is trained to push probabilities to 0 or 1.
- Capacity to memorize training data → overfit confidence to training distribution.
- BatchNorm / LayerNorm and many other architectural features push toward overconfident outputs.

This is well-documented for image classification ([Guo et al. 2017](https://arxiv.org/abs/1706.04599), "On Calibration of Modern Neural Networks") and is similar for transformers and LLMs.

## How to fix calibration

### Temperature scaling

After training, learn a single scalar T dividing logits before softmax: `p = softmax(z / T)`. T > 1 spreads probabilities; T < 1 sharpens. Surprisingly effective for 1-parameter fix.

### Platt scaling

A logistic regression on top of model outputs, often used for binary classification. Learns `sigmoid(a * f(x) + b)` where `f(x)` is the model output. Two parameters; calibrates well.

### Isotonic regression

A non-parametric monotonic regression of model outputs to true probabilities. Can fit more complex miscalibration patterns than temperature/Platt scaling, but needs more data to avoid overfitting.

### Label smoothing during training

Replace one-hot labels with `(1 - eps) * y + eps / K` (mass `eps` distributed across all classes). Trains the model to predict less peaky distributions; often improves calibration. `eps = 0.1` is a common default.

### Ensembling

Average predictions from multiple models. Often improves both accuracy and calibration. Expensive at inference time.

### Bayesian methods

Variational inference, Monte Carlo dropout, Bayesian neural networks. Give principled uncertainty estimates. Generally more complex than the alternatives; rarely worth it for production unless calibration is critical.

## Calibration for LLMs

LLMs are even worse-calibrated than typical classifiers, partly because:

- The probability over the next token is conditional on a long context; small differences cascade.
- RLHF / DPO training explicitly trades off calibration for helpfulness.
- LLMs are trained to produce confident-sounding outputs.

Production approaches:

- **Temperature is a sampling parameter, not a calibration parameter.** Setting `temperature=0` makes outputs deterministic but doesn't make probabilities calibrated.
- **Self-consistency**: sample N completions; the fraction that agree is a more reliable confidence signal than any individual probability score.
- **Verbalized confidence**: ask the model to also output a confidence score in natural language. Has been shown to be reasonably calibrated for some models, especially after specific fine-tuning.
- **Logprobs are noisy**: the raw token logprob is uncalibrated. Useful as a relative signal but not as a probability.

## What an interviewer expects you to say

If asked about calibration:

1. Define it precisely (predicted confidence = empirical accuracy).
2. Distinguish accuracy from calibration.
3. Mention that modern neural networks are typically overconfident.
4. Mention temperature scaling as the standard fix.
5. Bonus: discuss reliability diagrams, ECE, and label smoothing.

For LLM-team interviews specifically, mentioning that LLM confidence scores are unreliable and that self-consistency is more trustworthy is a senior signal.

## Common confusions

- **"My model has 95% accuracy so it's well calibrated."** Different things. A model with 95% accuracy can output `[0.99, 0.01]` for every example, in which case its 99%-confidence predictions are right only 95% of the time, overconfident.
- **"Temperature scaling is just for sampling."** In LLMs, "temperature" is a sampling parameter. In calibration, "temperature scaling" is a post-hoc calibration trick that scales logits. Same name, different mechanism.
- **"Calibration is for classification only."** Regression models also have calibration concepts (CRPS, prediction interval coverage). Less commonly discussed but matters in forecasting.
- **"Better calibration helps accuracy."** Doesn't necessarily. Accuracy and calibration are separate axes. You can improve calibration without improving accuracy and vice versa.

## Why interviewers ask

Calibration questions test:
1. Whether you understand probabilities vs scores.
2. Whether you've consumed model outputs in a downstream system (forces awareness of calibration issues).
3. Whether you've handled the "neural networks are overconfident" reality.
4. Whether you can think about a model as part of a system, not just as an isolated classifier.

A common follow-up: "When does calibration *not* matter?" The senior answer: when the downstream system only consumes the argmax (the highest-scoring class), calibration doesn't matter for the decision, only accuracy does. As soon as the system consumes the probability or makes threshold-based decisions, calibration matters.

---

*Related: [decision thresholds, asymmetric costs, and abstention](/concepts/decision-thresholds-asymmetric-costs-abstention/), [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), and [How do you handle hallucinations in production?](/questions/handle-hallucinations-in-production/).*
