---
title: "L1 vs L2 regularization, beyond the formula"
description: "The math is identical to most candidates: penalty terms in the loss. The senior signal is the Bayesian interpretation, the optimization geometry, and when each is the right choice."
date: "2026-05-07"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: ML breadth at every level.*

The L4 answer states the formulas. The L6 answer explains *why* L1 induces sparsity geometrically, names the Bayesian priors each corresponds to, and gives a clear practitioner rule for when to pick which.

## What an L4 answer sounds like

> "L1 adds the absolute value of weights to the loss, L2 adds the squared values. L1 leads to sparse solutions, L2 leads to small weights."

Correct, no depth. You've heard the rule, not the reason.

## What an L5 answer sounds like

> "Both add a penalty term to the loss to constrain the weights:
>
> - L1: `lambda * sum(|w_i|)`
> - L2: `lambda * sum(w_i^2)`
>
> The Bayesian view: L1 corresponds to a Laplace prior on the weights centered at zero; L2 corresponds to a Gaussian prior. Both pull weights toward zero, but with different shapes.
>
> Why L1 induces sparsity (the geometric view): the L1 ball has corners at the axes. The optimum of the loss + L1 penalty often lands at one of those corners, which means many weights are exactly zero. The L2 ball is round, so the optimum lands somewhere off the axes, with all weights small but rarely zero.
>
> Practical use: L2 (weight decay) is the default for deep networks; pick L1 when you want feature selection (sparse models, interpretability, downstream sparse computation)."

This is L5. You've named the prior interpretation, given the geometric intuition, and made a practitioner recommendation.

## What an L6 answer sounds like

> "...a few more things worth saying:
>
> **L1 is non-differentiable at zero**, which means standard gradient descent doesn't drive weights exactly to zero; you need proximal methods (ISTA, FISTA) or specialized solvers. In practice, frameworks use a sub-gradient (typically zero at zero) and rely on small numerical noise for actual sparsity.
>
> **AdamW vs Adam with weight decay**: 'L2 regularization' as a `+ lambda * w^2` term in the loss interacts badly with adaptive optimizers (the regularization gets scaled by the adaptive learning rate). AdamW decouples weight decay by applying it directly to the weights, not the gradient. This is what most modern transformers use.
>
> **Elastic Net** combines both. Useful when you want some sparsity but L1 alone is unstable in the presence of correlated features.
>
> **Implicit regularization** matters more than people think. SGD's noise, early stopping, and architecture (e.g., dropout, batch norm) often dominate explicit penalty regularization in deep learning. For very large models, weight decay is more about training stability than overfitting prevention."

## Tells that get you a strong-hire vote

- You name the **Bayesian priors** (Laplace for L1, Gaussian for L2).
- You give the **geometric** sparsity argument, not just the empirical claim.
- You distinguish **AdamW vs Adam with weight decay**.
- You acknowledge **implicit regularization** matters more for deep nets.

## Tells that get you down-leveled

- "L1 is sparse, L2 is small" with no further explanation.
- Confusion about the AdamW vs Adam weight-decay distinction.
- Suggesting L1 for deep networks as a default (rare; usually L2 or weight decay).
- No awareness that L1 needs special optimizers for true sparsity.

## Common follow-up

"You said weight decay is the default. When would you turn it off?"

The L6 answer:

> "Frontier-scale LLM pretraining uses very small weight decay (e.g., 0.01 or 0.1 in AdamW) because the model is underfitting at trillion-token scale, not overfitting. Some teams report no quality loss with weight decay = 0; others find a small WD helps with training stability. For typical supervised models on small data, default weight decay (1e-4 to 1e-2) is meaningful regularization."

---

*Related reference: [Regularization](/reference/regularization/), [Adam, AdamW, and the modern optimizer landscape](/reference/adam-and-adamw/).*
