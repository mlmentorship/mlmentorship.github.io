---
title: "Weight decay vs. L2 regularization"
description: "L2 adds ½λ‖θ‖² to the loss; weight decay shrinks θ multiplicatively at each step. They are equivalent under SGD but not under Adam. Which is why AdamW exists."
date: "2026-02-27"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

L2 regularization adds a penalty $\frac{\lambda}{2} \|\theta\|^2$ to the loss; weight decay multiplies parameters by $(1 - \eta \lambda)$ at each step. Under vanilla SGD they are mathematically equivalent. Under adaptive optimizers like Adam they are **not**. And the difference is large enough that AdamW [(Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) is now the default for transformer training.

## The two formulations

### L2 (penalty added to loss)

$$
L_\text{total}(\theta) = L_\text{data}(\theta) + \tfrac{\lambda}{2} \|\theta\|^2
$$

Gradient: $\nabla L_\text{total} = \nabla L_\text{data} + \lambda \theta$.
Update under SGD: $\theta_{t+1} = \theta_t - \eta (\nabla L_\text{data} + \lambda \theta_t)$
$\quad\quad\quad\quad\quad\quad\quad\,= (1 - \eta \lambda) \theta_t - \eta \nabla L_\text{data}$.

### Weight decay (multiplicative shrink)

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \nabla L_\text{data}
$$

Same expression. Under **vanilla SGD**, the two are identical.

## Why they diverge under Adam

Adam scales the gradient per parameter by $1/\sqrt{v_t + \varepsilon}$ before applying the update. The L2 contribution $\lambda \theta$ is part of the gradient, so it gets divided by $\sqrt{v_t + \varepsilon}$ too:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat m_t + \lambda \theta_t}{\sqrt{\hat v_t + \varepsilon}}
$$

The effective decay on each parameter is now scaled by $1/\sqrt{v_t}$. Large for parameters with small gradient variance, small for parameters with large gradient variance. This couples regularization to the gradient history in an unintended way.

**AdamW** decouples them: apply Adam to the data loss only, and then shrink the parameters multiplicatively as a separate step:

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t + \varepsilon}}
$$

The shrink term has no $1/\sqrt{v_t}$ scaling. This recovers the SGD-equivalent behavior.

## Empirical impact

Loshchilov & Hutter (2019) and many follow-up benchmarks show AdamW generalizes meaningfully better than Adam-with-L2 across vision and NLP. The exact gain depends on the task; on transformer LLM training the gap is large enough that essentially all modern training uses AdamW.

## What to skip

Common practice: do not decay biases, LayerNorm parameters, or embeddings. These are 1D parameters with different statistical roles, and decaying them often hurts. Standard implementations construct two parameter groups: `{decay: linear weights, conv kernels}` and `{no decay: biases, norms, embeddings}`.

## Common pitfalls

- **Using `Adam` with `weight_decay > 0`** in PyTorch. This applies L2-as-gradient, *not* AdamW. Use `AdamW` explicitly.
- **Decaying bias and LayerNorm parameters.** Hurts performance; exclude them via parameter groups.
- **Picking $\lambda$ from a CNN recipe.** $\lambda = 1\text{e-}4$ for ResNets; $\lambda = 0.1$ for AdamW transformer pretraining (with the no-decay carve-out). Different scale, different rule of thumb.
- **Forgetting that decay scales with LR.** Effective shrink per step is $\eta \lambda$. Halving LR halves effective decay; you may need to compensate.

## Related

- [Adam and AdamW](/reference/adam-and-adamw/). For the optimizer-side derivation.
- [Regularization](/reference/regularization/). Broader survey of regularization techniques.
