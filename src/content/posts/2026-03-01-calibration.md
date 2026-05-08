---
title: "Calibration: when your model says 80% it should be right 80% of the time"
description: "Accuracy isn't enough; you also want predictions to mean what they say. Calibration is the difference."
date: "2026-03-01"
draft: false
tags: ["concepts"]
category: "concepts"
---


## One-line definition

A model is well-calibrated if among predictions made with confidence p, the fraction that are correct is also p. A model can be accurate but poorly calibrated, or calibrated but inaccurate; both matter for production.

## Why it matters

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

*Related: [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), [How do you handle hallucinations in production?](/questions/handle-hallucinations-in-production/).*
