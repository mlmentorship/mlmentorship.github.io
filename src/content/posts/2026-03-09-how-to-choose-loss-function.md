---
title: "How do you choose a loss function?"
description: "The loss is the objective. Picking the wrong one means optimizing for the wrong thing, no matter how well you train. The senior answer derives the loss from the problem, not from a list."
date: "2026-03-09"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at every level.*

The L4 candidate names the canonical loss for the task type. The L6 candidate explains that the loss is a model of the noise distribution and the cost structure, and reasons from the problem to the loss.

## What an L4 answer sounds like

> "For classification, cross-entropy. For regression, MSE. For ranking, pairwise hinge. Use what fits the task."

Right inputs, no model. You've memorized which loss goes with which task type, but you can't reason about it.

## What an L5 answer sounds like

> "The loss encodes two things: the *noise model* of the targets and the *cost structure* of errors.
>
> - **Cross-entropy** for classification: assumes targets are samples from a categorical distribution, recovers MLE.
> - **MSE** for regression: assumes Gaussian noise on the targets, recovers MLE.
> - **MAE** for regression with heavy-tailed errors or median-targeting: assumes Laplace noise, more robust to outliers.
> - **Huber** for regression: a smooth interpolation between MSE (small errors) and MAE (large errors). Common when occasional outliers shouldn't dominate gradients.
> - **Pairwise / listwise losses** for ranking: encode the ordering of items, not absolute scores.
> - **Triplet / contrastive losses** for representation learning: encode that similar items should be close, dissimilar far.
>
> If the cost structure is asymmetric (e.g., false negatives much worse than false positives in fraud detection), I'd weight the loss accordingly or use a focal loss to focus on hard examples."

This is L5. You've connected losses to noise models and cost structures.

## What an L6 answer sounds like

> "...a few practical considerations:
>
> **The loss should be calibrated for the metric you actually care about.** A model trained on cross-entropy is calibrated for log-likelihood, not for AUC or precision at K. If your downstream metric is AUC, you might benefit from a ranking-aware loss; if it's precision at K, focal loss or top-K cross-entropy.
>
> **Multi-task losses are weighted sums of per-task losses.** The weighting matters and is hard to set; uncertainty-weighted multi-task loss (Kendall et al.) and gradient-magnitude balancing (GradNorm) are principled approaches.
>
> **For LLMs, the loss is usually next-token cross-entropy, but the *important* loss is the downstream behavior**. SFT trains on next-token CE; DPO replaces it with a preference loss; RLHF replaces it with a reward-model-derived gradient. Each shapes the model differently for the same evaluation goal.
>
> **Auxiliary losses can stabilize training without changing the main objective**: an auxiliary reconstruction loss for representation learning, an auxiliary load-balance loss for mixture-of-experts. These are tools to shape gradient flow, not just to add tasks."

## Tells that get you a strong-hire vote

- You connect the loss to the **noise model** (Gaussian / Laplace / categorical).
- You distinguish **mean** vs **median** targeting (MSE vs MAE).
- You bring up **calibration vs the downstream metric** as separate concerns.
- You mention **focal loss** or class weighting for asymmetric cost structures.

## Tells that get you down-leveled

- Memorized list with no underlying reasoning.
- Suggesting MSE for classification (vanishing gradients on confident-wrong predictions).
- No mention of cost asymmetry or imbalance.
- Treating loss selection as a closed problem.

## Common follow-up

"How would you train a model that values precision much more than recall?"

The L6 answer:

> "Several options, ranked by complexity. (1) Threshold tuning on the original model. (2) Cost-sensitive loss: weight false positives much more than false negatives in cross-entropy. (3) Train with focal loss to focus on hard examples. (4) Train two-stage: a high-recall first stage, then a high-precision filter. The right answer depends on whether the precision/recall asymmetry is large enough to need a structural change or small enough that threshold tuning suffices."

---

*Related: [Cross-entropy and softmax](/concepts/cross-entropy-softmax/), [Regularization](/concepts/regularization/), [Calibration](/concepts/calibration/).*
