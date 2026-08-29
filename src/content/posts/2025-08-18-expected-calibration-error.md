---
title: "Expected Calibration Error (ECE)"
description: "How well do predicted probabilities match empirical frequencies? Bin predictions by confidence, compare bin-mean confidence to bin-accuracy."
date: "2025-08-18"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Expected Calibration Error** measures how well a model's predicted probabilities match empirical accuracies. Bin predictions by predicted confidence, and compute the weighted average of $|\text{accuracy in bin} - \text{average confidence in bin}|$:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \cdot \big| \text{acc}(B_m) - \text{conf}(B_m) \big|.
$$

A classifier that scores well on accuracy can still produce wildly miscalibrated probabilities. Predicting "90% confident" when only 60% of such predictions are correct. Calibration matters whenever:

- The probability is used downstream (decision thresholds, expected-cost calculations, risk scoring).
- A human reads the probability (medical diagnosis, fraud alerts).
- The model is combined with other signals (Bayesian fusion).

ECE is the standard single-number calibration metric.

## The mechanism

For a binary classifier producing scores $p_i \in [0, 1]$ on $N$ examples with true labels $y_i$:

1. **Bin** the predictions by confidence. Standard: $M = 10$ equal-width bins covering $[0, 1]$.
2. For each bin $B_m$:
   - $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i$ (average predicted probability).
   - $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}[\hat y_i = y_i]$ (empirical accuracy).
3. **Aggregate**: ECE = weighted average of bin gaps.

A perfectly calibrated model has ECE = 0: every bin's empirical accuracy equals its average predicted confidence. Common modern deep classifiers have ECE 0.05–0.20. Predicted confidence is systematically inflated.

## Reliability diagram

The visual companion to ECE: plot bin accuracy vs. bin confidence. Perfect calibration is the diagonal $y = x$. Above-diagonal: under-confident. Below-diagonal: over-confident (the typical deep-net failure).

Always plot the reliability diagram alongside reporting ECE. Single ECE number can hide dramatic per-bin issues.

## Variants

- **Maximum Calibration Error (MCE)**: $\max_m |\text{acc}(B_m) - \text{conf}(B_m)|$. Worst-case bin gap.
- **Adaptive ECE**: equal-frequency bins instead of equal-width. Stable when predictions concentrate near 0 or 1.
- **Class-wise ECE**: per-class calibration; matters in multi-class.
- **Top-label ECE** (multi-class): compute ECE on the predicted-class probability only.

## Why deep nets are miscalibrated

Modern neural networks [(Guo et al., 2017)](https://arxiv.org/abs/1706.04599) are typically **overconfident**:

- Trained on cross-entropy, which keeps pushing logits toward $\pm \infty$ on training data.
- Standard regularization (weight decay, dropout) helps generalization but doesn't fix calibration.
- Modern architectures (ResNets, transformers) are more miscalibrated than older small models.

## Calibration methods

Post-hoc rescaling of predicted probabilities, learned on a held-out set:

- **Temperature scaling** [(Guo et al., 2017)](https://arxiv.org/abs/1706.04599): divide logits by a single learned scalar $T > 1$. Cheap; preserves accuracy; usually halves ECE. The default modern choice.
- **Platt scaling**: fit a logistic regression on the logits. Used historically with SVMs.
- **Isotonic regression**: fit a non-parametric monotonic mapping. More flexible; can overfit.
- **Vector / matrix scaling**: per-class temperature. More parameters; risk of overfitting if calibration set is small.

Calibration is **lossless on accuracy** (monotonic transformations preserve argmax). There's no reason not to do it.

## Limits of ECE

- **Binning artifact**: ECE depends on bin choice. Adaptive ECE is more stable.
- **Confidence vs. probability of correctness**: ECE on multi-class typically uses only the top-class probability, ignoring whether the full distribution is well-calibrated.
- **Doesn't measure sharpness**: a model that predicts $0.5$ for everything has ECE = 0 if base rate is $0.5$, but is useless. Combine ECE with proper scoring rules (Brier score, log loss).

## Common pitfalls

- **Computing ECE on training data.** Always use held-out data.
- **Reporting ECE without a reliability diagram.** Visual asymmetries can be invisible in the single number.
- **Skipping calibration in production.** Temperature scaling is one line of code and often halves ECE.
- **Confusing calibration with accuracy.** A 95%-accuracy model with ECE 0.20 is still untrustworthy whenever the probability matters.

## Related

- [Calibration](/concepts/calibration/). Broader treatment of calibration methods.
- [Confusion matrix](/concepts/confusion-matrix-and-classification-metrics/). Accuracy-based metrics.
