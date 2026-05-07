---
title: "Label smoothing"
description: "Replace one-hot targets with a softened distribution that puts ε mass on the wrong classes. Improves calibration, sometimes hurts retrieval."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Label smoothing [(Szegedy et al., 2016)](https://arxiv.org/abs/1512.00567) replaces the hard one-hot target $y$ with $\tilde{y} = (1 - \varepsilon) \cdot y + \varepsilon / K$, where $K$ is the number of classes and $\varepsilon$ is a small smoothing constant (typically 0.1). The cross-entropy loss is computed against $\tilde{y}$.

## Why it matters

With one-hot targets, the cross-entropy loss is unbounded as the model becomes confident. It can always reduce loss further by making the correct logit larger. This pushes the model toward arbitrarily large logits and overconfident predictions, which are poorly calibrated.

Label smoothing caps how much loss can be reduced by confidence and forces the model to maintain non-zero probability on incorrect classes. The effects:

- **Better calibration**: predicted probabilities track empirical accuracy more closely.
- **Slightly better generalization** on most classification benchmarks.
- **Standard in transformer training**: original "Attention Is All You Need" used $\varepsilon = 0.1$; LLM pretraining occasionally uses it.

## The mechanism

For a classification problem with $K$ classes and true class $c$:

- Hard target: $y_i = 1$ if $i = c$, else 0.
- Smoothed target: $\tilde{y}_i = 1 - \varepsilon + \varepsilon/K$ if $i = c$, else $\varepsilon/K$.

Cross-entropy with smoothed targets:

$$
L = -\sum_i \tilde{y}_i \log p_i = -(1 - \varepsilon) \log p_c - \frac{\varepsilon}{K} \sum_i \log p_i.
$$

The first term is the standard cross-entropy; the second is an entropy-like penalty that pulls the predicted distribution toward uniform.

Equivalent view: the optimal $p_c$ for label smoothing is $1 - \varepsilon + \varepsilon/K$, not 1. The model has no incentive to push the correct logit beyond what produces this target probability.

## When to use

- **Language modeling** (transformer training): standard $\varepsilon = 0.1$.
- **Image classification** with hard labels: standard, $\varepsilon = 0.1$.
- **Distillation**: not needed; the teacher's soft targets already provide the regularization.
- **Retrieval / contrastive learning**: usually skipped; sharp distributions are sometimes needed for good top-1.

## Side effects

- **Calibration improves**: temperature 1 softmax becomes closer to actual confidence.
- **Top-1 accuracy roughly unchanged or marginally improved**.
- **Worse for retrieval / nearest-neighbor**: the embeddings cluster less tightly because the model is penalized for confidence (Müller et al., 2019).
- **Worse for distillation as teacher**: a label-smoothed teacher provides less informative soft targets.

## Common pitfalls

- **Stacking with mixup / cutmix.** These already softening targets; adding label smoothing on top double-counts.
- **Using on a regression problem.** Label smoothing is for categorical cross-entropy; it has no meaning for MSE.
- **Choosing $\varepsilon$ too large.** $\varepsilon = 0.5$ destroys signal; $\varepsilon = 0.1$ is the universal default.
- **Forgetting to disable for eval-only metrics.** Loss numbers with label smoothing are not directly comparable to one-hot loss numbers.
