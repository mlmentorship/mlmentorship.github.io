---
title: "Value-based vs. policy-based RL"
description: "Two paradigms in reinforcement learning. Value-based learns Q(s, a) and acts greedily; policy-based directly parametrizes the policy. When to use which."
date: "2026-04-22"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

**Value-based** RL learns a value function (typically $Q(s, a)$) and acts greedily with respect to it: $\pi(s) = \arg\max_a Q(s, a)$. **Policy-based** RL directly parametrizes a stochastic policy $\pi_\theta(a \mid s)$ and optimizes $\theta$ via the policy gradient. **Actor-critic** combines both.

## Why it matters

Choosing the right paradigm is a central decision in RL system design. Mismatch leads to poor sample efficiency, instability, or simply not working. For instance, value-based methods are awkward in continuous action spaces, and pure policy-based methods are sample-inefficient in tabular settings.

## Value-based methods

Examples: Q-learning, DQN, Rainbow, distributional Q-learning.

**Strengths:**

- **Off-policy data reuse**: replay buffers enable training on old data, large effective sample size.
- **Lower variance** than vanilla policy gradient.
- **Good for discrete action spaces** with moderate cardinality.

**Weaknesses:**

- **Continuous actions** require $\arg\max_a Q(s, a)$. A separate optimization at each step. DDPG / TD3 learn a deterministic actor for this.
- **No stochastic policies**: greedy w.r.t. $Q$ is deterministic. For exploration, must add $\varepsilon$-greedy externally.
- **Maximization bias**: $\max_a Q$ overestimates true $Q$ when $Q$ is noisy.
- **Many tricks needed** for stable deep variants: target networks, prioritized replay, double DQN.

**Use when:**

- Discrete actions, environment can be simulated cheaply (Atari, board games).
- Need to reuse offline data.
- Value function approximation is structurally easy (e.g., low-d state).

## Policy-based methods

Examples: REINFORCE, A2C, A3C, TRPO, PPO.

**Strengths:**

- **Continuous actions** are natural: parametrize $\pi$ as Gaussian or similar.
- **Stochastic policies** built-in: useful for exploration, mixed-strategy equilibria, partial observability.
- **Direct objective**: maximize expected return.
- **Simpler theoretical framing**: no Bellman equations, just expectation gradients.

**Weaknesses:**

- **Sample inefficient**: standard policy gradient is on-policy. Discard data after each gradient step.
- **High variance**: Monte Carlo gradient estimator is noisy without baselines.
- **Local optima**: policy gradient can get stuck in deterministic suboptimal policies.

**Use when:**

- Continuous control (robotics, physical simulation).
- Stochastic policy needed (exploration, multi-agent).
- Policy is naturally differentiable but value function is not (e.g., LLMs as policies in RLHF).

## Actor-critic: combining the two

An **actor-critic** algorithm trains both:

- **Actor** $\pi_\theta(a \mid s)$: the policy.
- **Critic** $V_\phi(s)$ or $Q_\phi(s, a)$: the value function, used as a baseline / target for the actor's gradient.

The critic reduces the variance of the policy gradient; the actor handles continuous actions cleanly. Almost all modern RL algorithms (PPO, SAC, DDPG, TD3, IMPALA) are actor-critic.

## Decision matrix

| Problem | First choice |
|---------|------------|
| Discrete actions, abundant simulation | DQN / Rainbow |
| Continuous control, sample-efficient | SAC |
| Continuous control, simple and robust | PPO |
| LLM alignment | PPO or DPO |
| Multi-agent | MAPPO, IMPALA |
| Real-world robotics with limited samples | SAC or model-based RL |
| Board games / planning | AlphaZero-style (MCTS + learned policy/value) |
| Partially observable, RNN policy | PPO with LSTM/transformer policy |

## What about model-based RL?

A third paradigm: learn a dynamics model $p(s' \mid s, a)$ and plan with it. Examples: Dreamer, MuZero, World Models. **Strengths**: extreme sample efficiency. **Weaknesses**: dynamics model errors compound; engineering complexity. Used when real-world samples are expensive (robotics, healthcare).

## Common pitfalls

- **Choosing value-based for continuous control.** Awkward; SAC or DDPG-family if you must use Q-learning.
- **Choosing pure policy gradient for sample-rich discrete problems.** DQN-family is much more sample-efficient.
- **Treating actor-critic as fundamentally different.** It's just policy gradient with a learned baseline; understand both pieces.
- **Ignoring variance in policy gradient.** Without baselines + advantage normalization, you'll see noisy curves and slow learning.

## Related

- [Q-learning](/reference/q-learning/). Canonical value-based.
- [Policy gradient](/reference/policy-gradient/). Canonical policy-based.
- [PPO](/reference/ppo/). Modern actor-critic.
