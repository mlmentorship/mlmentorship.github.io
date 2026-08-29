---
title: "Policy gradient methods"
description: "Directly optimize the policy by following the gradient of expected return. REINFORCE, actor-critic, and the foundation of modern RL."
date: "2025-11-20"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Policy gradient** methods directly parametrize a policy $\pi_\theta(a \mid s)$ and optimize $\theta$ by ascending the gradient of expected return:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R_t \right].
$$

This is the **policy gradient theorem** [(Sutton et al., 1999)](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html).

Policy gradient methods are the foundation of modern continuous-control RL (SAC, PPO), of large-scale RL (AlphaGo's policy net, OpenAI Five), and of LLM alignment (RLHF uses PPO). They are essential whenever:

- The action space is continuous (no easy $\max_a$).
- A stochastic policy is desirable (exploration, multi-modal optimal policies).
- You can write a clean, differentiable policy parametrization.

## REINFORCE

The simplest policy gradient [(Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696):

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return-to-go from step $t$.

Sample a trajectory by running the policy. For each step, compute $\log \pi_\theta(a_t \mid s_t)$ and weight by the actual return. Backprop. Take a gradient step.

**Pros**: extremely general. Works for any differentiable policy.
**Cons**: high variance. Return $G_t$ is a noisy estimate of expected return; gradient estimator dominates by random rollout luck.

## Variance reduction: baselines

Subtract any state-dependent baseline $b(s_t)$ from $G_t$. Does not change the expected gradient (so no bias) but can drastically reduce variance:

$$
\nabla_\theta J = \mathbb{E}\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right].
$$

Optimal baseline: a value function $V(s_t)$. This leads directly to actor-critic.

## Actor-critic

Two networks:

- **Actor** $\pi_\theta(a \mid s)$. The policy.
- **Critic** $V_\phi(s)$. Value function estimator.

Use the critic as the baseline. Define the **advantage** $A(s, a) = Q(s, a) - V(s) \approx G_t - V_\phi(s_t)$. Update:

$$
\nabla_\theta J = \mathbb{E}\!\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \right].
$$

Critic is trained by TD: $V_\phi(s_t) \leftarrow V_\phi(s_t) + \alpha [r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)]$.

This is the basic **A2C** / **A3C** algorithm.

## Generalized Advantage Estimation (GAE)

Instead of one-step or full-return advantages, use a weighted sum:

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

$\lambda \in [0, 1]$ trades bias for variance. $\lambda = 0$ → one-step TD. $\lambda = 1$ → Monte-Carlo. Standard $\lambda = 0.95$ in PPO.

## Trust regions and PPO

Naive policy gradient updates can drastically change the policy in one step, leading to collapse. **Trust Region Policy Optimization** (TRPO; [Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) constrains the KL divergence between old and new policies. **PPO** [(Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) replaces the constraint with a clipped surrogate objective:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}\!\left[ \min\big( r_t(\theta) A_t,\; \mathrm{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) A_t \big) \right]
$$

with $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ and $\varepsilon = 0.2$ typical.

PPO is the dominant policy gradient algorithm in 2026: simpler than TRPO, robust, well-understood, used in RLHF.

## Off-policy actor-critic

For sample efficiency in continuous control, use a replay buffer with importance sampling correction or Q-function critics. **DDPG**, **TD3**, **SAC** are the standard off-policy actor-critic algorithms. SAC adds a maximum-entropy bonus that encourages exploration.

## Common pitfalls

- **High variance.** Always use a baseline; always normalize advantages within a batch (subtract mean, divide by std).
- **No entropy bonus.** Policies collapse to deterministic; add $-\beta \cdot H(\pi_\theta)$ to the loss to encourage exploration.
- **Reusing samples without importance correction.** On-policy methods (REINFORCE, PPO) sample from the current policy; using stale samples introduces bias. PPO's clipping caps the staleness.
- **Mismatch between rollout and learning.** GPU-driven RL often has the policy version drift between rollout and gradient update; PPO's IS ratio handles small mismatches.
- **Treating policy gradient as universally low-variance.** It isn't. Q-learning often has lower variance for discrete-action problems.

## Related

- [Q-learning](/concepts/q-learning/). Value-based alternative.
- [RLHF and DPO](/concepts/rlhf-and-dpo/). PPO applied to LLM alignment.
- [PPO](/concepts/ppo/). Algorithm-level details.
