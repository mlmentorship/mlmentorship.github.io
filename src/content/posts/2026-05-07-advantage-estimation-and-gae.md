---
title: "Advantage estimation and GAE"
description: "Policy gradients need a low-variance estimate of how much better an action was than average. GAE is the standard answer: an exponentially weighted blend of n-step returns."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

The **advantage** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ measures how much better action $a$ is than the policy's average. **Generalized Advantage Estimation** (GAE, [Schulman et al., 2016](https://arxiv.org/abs/1506.02438)) estimates it as an exponentially weighted average of $n$-step TD residuals, controlled by a single parameter $\lambda$.

## Why it matters

Policy gradient methods optimize $\nabla_\theta J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a \mid s) \cdot \Psi]$. The choice of $\Psi$ controls the bias-variance tradeoff:

- $\Psi = R$ (full return): unbiased, high variance.
- $\Psi = Q^\pi(s, a)$: lower variance but needs an action-value estimator.
- $\Psi = A^\pi(s, a)$: same expectation as $Q$ but with the baseline subtracted, lower variance.

Substituting an estimator for $A$ introduces bias. GAE makes this tradeoff explicit and tunable. It is the default advantage estimator in PPO, the most widely deployed RL algorithm.

## The mechanism

Define the **TD residual** at step $t$:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

The $n$-step advantage estimate is

$$
\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}.
$$

GAE blends all $n$-step estimates with exponential weight $\lambda$:

$$
\hat{A}_t^{\text{GAE}}(\gamma, \lambda) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}.
$$

In code this collapses to a backward recursion:

$$
\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}.
$$

A single backward pass over the trajectory.

## The two knobs

- **$\gamma$ (discount)**: how much future reward matters. Part of the problem definition; usually 0.99 for episodic, 0.95 to 0.999 for continuing tasks.
- **$\lambda$ (GAE)**: bias-variance dial.
  - $\lambda = 0$ recovers the 1-step TD residual: low variance, biased by $V$ errors.
  - $\lambda = 1$ recovers the full Monte Carlo return minus $V(s_t)$: unbiased, high variance.
  - $\lambda = 0.95$ to $0.97$ is the standard for PPO.

## Why subtracting a baseline reduces variance

For any function $b(s)$ depending only on state, $\mathbb{E}_\pi[\nabla \log \pi(a \mid s) \cdot b(s)] = 0$. So the gradient estimator

$$
\nabla \log \pi(a \mid s) \cdot (Q^\pi(s, a) - b(s))
$$

has the same expectation but lower variance, when $b$ correlates with $Q^\pi$. The optimal baseline is exactly $V^\pi$, hence the advantage formulation.

## How it is used in PPO

PPO trains an actor and a value critic jointly. At each rollout:

1. Run the policy for $T$ steps in $N$ parallel environments. Collect transitions.
2. Run the value network on every observed state to get $V(s_t)$.
3. Compute $\delta_t$ and then $\hat{A}_t$ via the GAE recursion.
4. Compute returns as $\hat{R}_t = \hat{A}_t + V(s_t)$ for the value-function regression target.
5. Normalize advantages (subtract mean, divide by std) per batch. Important for training stability.
6. Train policy with the clipped objective using $\hat{A}_t$, train value network on $\hat{R}_t$.

## Common pitfalls

- **Forgetting to bootstrap on truncation.** When an episode is cut off mid-trajectory (not because of termination), $\delta_T$ should use $V(s_T)$ as the bootstrap. Conflating truncation with termination is a frequent bug.
- **Not normalizing advantages.** PPO almost always benefits from per-batch advantage normalization.
- **Using $\lambda = 1$ with a large $\gamma$ on long horizons.** Variance explodes.
- **Applying GAE in off-policy settings without correction.** GAE assumes on-policy data. With a replay buffer and importance sampling, V-trace or Retrace is the corrected version.

## Related

- [Proximal Policy Optimization](/concepts/proximal-policy-optimization-ppo/).
- [Policy gradient methods](/concepts/policy-gradient-methods/).
- [Actor-critic methods](/concepts/actor-critic-methods/).
