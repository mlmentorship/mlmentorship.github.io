---
title: "Regularization: L1, L2, dropout, early stopping, and the modern view"
description: "The classical regularizers + the modern reality that SGD's noise is itself a regularizer. The hierarchy of choices when your model is overfitting."
date: "2026-01-30"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Regularization constrains effective model capacity to reduce overfitting. It includes explicit forms (L1/L2, dropout, early stopping, data augmentation) and implicit forms (SGD noise, optimizer choice, architecture).

Overfitting is the most common failure mode of moderately-sized models on moderately-sized datasets. Regularization is the response. Modern large-scale models often don't need explicit regularization (they're underfitting at trillion-token scale), but for almost everything outside frontier LLM pretraining, regularization choices matter.

## The classical lineup

### L2 regularization (weight decay)

Add `lambda * ||theta||^2` to the loss. Penalizes large weights.

- Bayesian interpretation: a Gaussian prior on the weights centered at zero.
- Effect: weights shrink toward zero; model preferentially uses many small weights instead of few large ones.
- Default `lambda` is typically 1e-4 to 1e-2 for SGD; smaller for Adam (1e-2 in AdamW for transformers).
- **Important**: in AdamW, weight decay is decoupled from the gradient (applied directly to weights). In standard Adam with `weight_decay > 0`, the weight decay gets scaled by the adaptive learning rate, which is usually wrong.

### L1 regularization

Add `lambda * ||theta||_1` to the loss. Penalizes sum of absolute values.

- Bayesian interpretation: a Laplace prior on the weights.
- Effect: induces *sparsity*, many weights become exactly zero. Useful when feature selection is desired.
- Less common in deep learning; more common in classical ML (LASSO).

### L1 + L2 (Elastic Net)

Combination of both. Sparsity from L1 + stability from L2. Common in classical ML, rarely used in deep learning.

### Dropout

Randomly zero a fraction `p` of activations during training; scale by `1/(1-p)` during training so the expected output is unchanged at test time (inverted dropout).

- Why it works (three frames): regularization (prevents co-adaptation), implicit ensembling (averaging exponentially many sub-networks), Bayesian approximation (variational inference over weights).
- Default `p = 0.1` to `0.3` for most networks; lower (0.0-0.1) for very large models.
- **In transformers**: usually applied to attention weights, attention output, and FFN intermediate. Not on residual stream, not on layer norms, not on embedding lookup.
- **Modern LLMs often use no dropout** during pretraining because they're not overfitting at scale. SFT and RLHF stages may add small dropout.

### Early stopping

Monitor validation loss; stop training when it stops improving for `N` epochs (or steps).

- Implicitly limits model capacity by limiting the number of update steps.
- Equivalent to L2 regularization in some idealized settings (Goodfellow, 7.8).
- Cheap, effective, hard to mess up. Should be standard in any training pipeline that doesn't run to a fixed budget.

### Data augmentation

Apply label-preserving transformations to inputs during training. Random crops, flips, color jitter for images; back-translation, synonym replacement for text; SpecAugment for audio.

- Effectively multiplies your data by the number of augmentations.
- One of the highest-ROI regularization techniques when applicable.
- For LLMs: not used in the classical sense, but data mixing strategies serve a similar role.

## The modern view: implicit regularization

A surprising 2017-2020 result: SGD itself is a regularizer. The noise from mini-batch sampling biases SGD toward "flat" minima that generalize better than the sharper minima found by full-batch optimization.

Consequences:

- Smaller batch sizes can generalize better than larger ones (when controlling for compute), because the gradient noise is larger.
- Adam's adaptivity reduces this implicit regularization, which is part of why SGD sometimes generalizes better than Adam in some settings.
- Momentum and weight decay interact non-trivially with this; the optimal weight decay value is different in different optimizers and at different batch sizes.

For modern LLM-scale training, the picture is more complicated. The implicit regularization framing is most useful for moderately-sized supervised models.

## When to use what

A practical hierarchy:

1. **Are you overfitting?** Train accuracy &gt;&gt; validation accuracy. If yes, regularize.
2. **Is your dataset small?** First try data augmentation (often biggest win) and dropout.
3. **Are your weights large or unstable?** Add L2 (AdamW with `weight_decay`).
4. **Are you stopping too late?** Add early stopping.
5. **Do you want sparsity?** Add L1 (rare in DL).
6. **Are you in the LLM-pretraining regime?** Skip dropout; weight decay still helps; data is the regularizer.

## Common confusions

- **"L1 = sparse, L2 = small."** Roughly right. L1 induces hard zeros; L2 shrinks but rarely to zero.
- **"Dropout in inference."** Standard recipe is OFF at inference. *Monte Carlo dropout* (keeping it ON to sample from the implicit posterior) is a separate technique for uncertainty estimation, not a default.
- **"More regularization = better generalization."** No. Too much regularization → underfitting. The right amount is task-dependent and must be tuned.
- **"Use weight decay AND dropout AND L2 etc."** Stacking many regularizers is fine but rarely necessary. Pick one or two that target your actual problem.

## Why interviewers ask

Regularization questions test:
1. Whether you know multiple techniques and can choose between them.
2. Whether you understand the Bayesian interpretations.
3. Whether you've kept up with the modern view (implicit regularization, the LLM-scale "regularization is data" view).
4. Whether you know the per-technique gotchas (AdamW vs Adam+weight_decay, dropout placement in transformers).

---

*Related: [Why does dropout work?](/questions/why-does-dropout-work/), [Adam, AdamW, and modern optimizer choices](/concepts/adam-and-adamw/).*
