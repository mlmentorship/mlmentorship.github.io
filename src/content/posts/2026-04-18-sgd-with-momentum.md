---
title: "SGD with momentum"
description: "Add a moving average of past gradients to the update. Smoother trajectories, faster convergence in narrow valleys, and the foundation of Adam's first moment."
date: "2026-04-18"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

SGD with momentum maintains a velocity $v_t = \beta v_{t-1} + g_t$ (an exponential moving average of past gradients) and updates parameters with $\theta_{t+1} = \theta_t - \eta v_t$ instead of with the raw gradient. Typical $\beta = 0.9$.

## Why it matters

Vanilla SGD bounces around in narrow loss valleys: gradients perpendicular to the valley axis cancel slowly, gradients along the axis are small. Momentum accumulates the consistent along-axis component while perpendicular components average to zero.

Empirically, momentum is essential for SGD to be competitive with adaptive optimizers on most problems. SGD without momentum is rarely used in modern training. Adam's first-moment estimate $m_t$ is essentially momentum, which is why Adam inherits this benefit.

## Two formulations

### Classical momentum (Polyak, 1964)

$$
v_t = \beta v_{t-1} + g_t \\
\theta_{t+1} = \theta_t - \eta v_t
$$

Effective LR for a constant gradient: $\eta / (1 - \beta)$. With $\beta = 0.9$, that's $10\eta$, so reducing $\beta$ effectively reduces the step size.

### Nesterov momentum (Nesterov, 1983)

Compute the gradient at the *look-ahead* position $\theta_t - \eta \beta v_{t-1}$ instead of at $\theta_t$. Updates:

$$
v_t = \beta v_{t-1} + \nabla L(\theta_t - \eta \beta v_{t-1}) \\
\theta_{t+1} = \theta_t - \eta v_t
$$

In practice, only marginally better than classical momentum on most workloads. Used in some vision training (e.g., ResNet original paper).

## Picking $\beta$

| Workload | $\beta$ |
|----------|---------|
| ResNet / CNN training | 0.9 |
| Reinforcement learning policy nets | 0.9 |
| Very noisy gradients (RL, contrastive) | 0.95 or 0.99 |
| Small batch with high noise | lower (0.5–0.8) to track recent gradients |

$\beta$ controls the effective averaging window: $1/(1-\beta)$ steps. $\beta = 0.9$ averages over ~10 steps; $\beta = 0.99$ averages over ~100.

## When SGD+momentum vs. Adam

| Situation | Default |
|-----------|--------|
| Vision (CNN, ViT) classification with strong regularization | SGD + momentum + cosine LR |
| Transformers (NLP, LLM training) | Adam / AdamW |
| Small datasets, fine-tuning | Adam (less hyperparameter tuning needed) |
| Sparse gradients (recsys embeddings) | Adam (per-parameter adaptive LR) |
| Reinforcement learning | Adam (default in PPO / DQN implementations) |

SGD+momentum often generalizes slightly better than Adam at convergence (sharp vs. flat minima discussion); Adam converges faster initially. For LLMs, Adam wins because the gradient distribution across parameters is highly non-uniform.

## Common pitfalls

- **Forgetting that momentum scales the effective LR.** Switching from $\beta = 0.9$ to $\beta = 0.99$ effectively multiplies the LR by 10.
- **Initializing $v_0 = 0$ without bias correction.** Early steps have biased velocity; for SGD this is usually fine, but Adam explicitly bias-corrects.
- **Mixing momentum across LR changes.** When the LR jumps, momentum carries old-LR-scaled velocities. Some implementations zero momentum at LR transitions.
- **Using SGD without momentum.** Almost always strictly worse; pick momentum or use Adam.
