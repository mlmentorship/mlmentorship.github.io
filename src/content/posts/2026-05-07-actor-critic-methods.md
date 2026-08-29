---
title: "Actor-critic methods"
description: "Policy gradient with a learned value baseline. The actor picks actions; the critic estimates how good they were. The architecture under PPO, A3C, SAC, and most modern RL."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An **actor-critic** algorithm trains two networks jointly: an **actor** $\pi_\theta(a \mid s)$ that chooses actions and a **critic** $V_\phi(s)$ (or $Q_\phi(s, a)$) that estimates expected return. The actor is updated with a policy gradient using the critic's estimate as a baseline; the critic is updated to fit observed returns.

Pure policy gradient (REINFORCE) is unbiased but high-variance. Pure value-based methods (Q-learning, DQN) are sample-efficient but only support discrete actions and struggle with stochastic optimal policies. Actor-critic combines both: low-variance gradient estimates from the critic, direct policy parameterization from the actor.

Almost every modern continuous-control RL algorithm is actor-critic: PPO, A2C/A3C, SAC, TD3, DDPG. RLHF for LLMs is actor-critic.

## The two updates

### Actor (policy gradient)

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot \hat{A}(s, a) \right],
$$

where $\hat{A}$ is an advantage estimate from the critic. See [GAE](/concepts/advantage-estimation-and-gae/) for the standard recipe.

### Critic (value regression)

$$
\mathcal{L}_\phi = \mathbb{E}\left[ (V_\phi(s_t) - \hat{R}_t)^2 \right],
$$

where $\hat{R}_t$ is the return target (TD($\lambda$) target, or the GAE-derived $\hat{A}_t + V_\phi(s_t)$).

Both gradients flow on the same data. Many implementations share most of the backbone between actor and critic and split off two heads at the end.

## The taxonomy

### A2C / A3C

Synchronous (A2C) or asynchronous (A3C) advantage actor-critic. Multiple environments collect rollouts in parallel; updates use $n$-step returns. Simple, robust, the historical baseline ([Mnih et al., 2016](https://arxiv.org/abs/1602.01783)).

### PPO

Adds a **clipped surrogate objective** to bound how much the new policy can move from the old one. Allows multiple epochs of update on the same rollout. The default for most practical RL today, including RLHF.

### DDPG

Off-policy deterministic actor-critic for continuous control ([Lillicrap et al., 2016](https://arxiv.org/abs/1509.02971)). The actor outputs a deterministic action; the critic is $Q_\phi(s, a)$; gradient flows from $Q$ back through the actor via the chain rule. Notoriously brittle.

### TD3

DDPG plus three fixes ([Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)): twin Q-networks (take the min to reduce overestimation), delayed policy updates, target policy smoothing. Much more stable than DDPG.

### SAC

Soft Actor-Critic ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)). Adds an entropy bonus to the reward, learning a maximum-entropy policy. Sample-efficient and robust; the standard for off-policy continuous control.

## On-policy vs off-policy

- **On-policy** (A2C, PPO): the data must come from the current policy. Rollouts are discarded after a few updates.
- **Off-policy** (DDPG, TD3, SAC): the data can come from any policy, stored in a replay buffer. Importance sampling or deterministic gradients handle the off-policyness.

Off-policy is more sample efficient (data can be reused) but harder to stabilize. On-policy is simpler and more reliable, at the cost of needing fresh rollouts.

## How RLHF fits

RLHF is just PPO (an actor-critic algorithm) with:

- The actor initialized from a pretrained LLM.
- The critic estimating value from the same model.
- The reward coming from a learned reward model trained on human preferences.

Modern alternatives (DPO, IPO, KTO) bypass the actor-critic step entirely; they reformulate the optimization as a supervised loss on preference pairs. Faster, simpler, and the new dominant approach in 2024 to 2026.

## Common pitfalls

- **Forgetting to detach the value estimate when computing the policy gradient.** The advantage flows into the actor; the actor gradient should not flow through the critic.
- **Sharing too much of the backbone**. With one body and two heads, the value loss can dominate the gradient and hurt the policy. Tune the loss weight or use separate networks for hard tasks.
- **Off-policy updates without correction**. PPO assumes near-on-policy data; running it with a deep replay buffer breaks the clipping bound.
- **Using PPO defaults blindly**. PPO has a dozen hyperparameters (clip range, GAE lambda, value-loss coef, entropy coef, learning rate, batch size, epochs per rollout). The interactions matter.

## Related

- [Policy gradient methods](/concepts/policy-gradient/).
- [Proximal Policy Optimization](/concepts/ppo/).
- [Advantage estimation and GAE](/concepts/advantage-estimation-and-gae/).
