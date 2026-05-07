---
title: "Q-learning"
description: "Learn the action-value function Q(s, a) by Bellman backups. The foundation of value-based RL. DQN, Rainbow, and the original Atari breakthroughs."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

**Q-learning** [(Watkins, 1989)](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) is an off-policy temporal-difference algorithm that learns the optimal action-value function $Q^*(s, a)$. The expected return from taking action $a$ in state $s$ and then acting optimally. By iterating the Bellman optimality update:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big].
$$

## Why it matters

Q-learning is the canonical value-based RL algorithm. Combined with deep neural networks (DQN; [Mnih et al., 2015](https://www.nature.com/articles/nature14236)), it produced the original deep RL breakthroughs on Atari and remains the foundation of value-based methods. Knowing Q-learning is the prerequisite for understanding: target networks, experience replay, double DQN, dueling networks, and the relationship to actor-critic methods.

## The setup

A Markov decision process (MDP): states $s$, actions $a$, transition $p(s' \mid s, a)$, reward $r(s, a)$, discount $\gamma \in [0, 1]$.

The **optimal action-value function**:

$$
Q^*(s, a) = \mathbb{E}\!\left[ \sum_{t=0}^{\infty} \gamma^t r_t \,\Big|\, s_0 = s, a_0 = a, \pi^* \right].
$$

The optimal policy: $\pi^*(s) = \arg\max_a Q^*(s, a)$.

## Bellman optimality

$Q^*$ satisfies the **Bellman optimality equation**:

$$
Q^*(s, a) = \mathbb{E}_{s'}\!\left[ r + \gamma \max_{a'} Q^*(s', a') \right].
$$

Q-learning approximates this by sampling: take action $a$, observe $r$ and $s'$, update $Q$ toward the target $r + \gamma \max_{a'} Q(s', a')$.

## Tabular Q-learning

For small finite state-action spaces, store $Q$ as a table. Sample transitions $(s, a, r, s')$, apply the update with learning rate $\alpha$. Guaranteed to converge to $Q^*$ if every state-action pair is visited infinitely often and learning rates satisfy Robbins-Monro conditions.

**Off-policy**: the update uses $\max_{a'} Q(s', a')$ regardless of which action is actually taken next. This decouples exploration policy (e.g., $\varepsilon$-greedy) from the learned greedy policy.

## Deep Q-Networks (DQN)

For large state spaces, replace the table with a neural network $Q_\theta(s, a)$. Two essential tricks make this work:

1. **Experience replay**: store transitions in a buffer, sample mini-batches uniformly. Breaks temporal correlation; stabilizes training; enables data reuse.
2. **Target network**: maintain a separate frozen copy $Q_{\theta^-}$ for the target $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$. Update $\theta^-$ to $\theta$ every $K$ steps. Prevents the target from chasing itself.

The DQN loss:

$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\!\left[ \big( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \big)^2 \right].
$$

## Variants

- **Double DQN** [(van Hasselt 2015)](https://arxiv.org/abs/1509.06461): use $\arg\max$ from online network, value from target network. Reduces overestimation bias.
- **Dueling DQN** [(Wang 2016)](https://arxiv.org/abs/1511.06581): factor $Q(s, a) = V(s) + (A(s, a) - \bar A(s))$.
- **Prioritized replay** [(Schaul 2015)](https://arxiv.org/abs/1511.05952): sample transitions with high TD error more often.
- **Rainbow** [(Hessel 2018)](https://arxiv.org/abs/1710.02298): combines six improvements; canonical strong baseline.
- **Distributional RL** (C51, IQN): predict the *distribution* of returns, not just the mean.

## Limitations

- **Maximization bias**: $\max_a Q(s, a)$ is biased upward when $Q$ is noisy. Double DQN partially fixes.
- **Continuous action spaces**: $\max_a$ becomes a non-trivial optimization; use deterministic policy gradients (DDPG, TD3, SAC) instead.
- **Sample efficiency**: deep Q-learning needs millions of environment steps; impractical for slow simulators.
- **Off-policy correction**: off-policy data can be biased; DQN papers often need careful replay buffer management.

## Q-learning vs. policy gradient

| Method | Q-learning | Policy gradient |
|--------|-----------|----------------|
| Learns | $Q(s, a)$ | $\pi(a \mid s)$ |
| Policy | Implicit ($\arg\max$) | Explicit |
| On/off policy | Off-policy | Usually on-policy |
| Continuous actions | Hard ($\max_a$) | Natural |
| Variance | Lower | Higher |
| Sample efficiency | Higher (data reuse) | Lower |

In practice for continuous control: SAC (combines Q-learning with stochastic policy). For discrete actions with small action space: DQN-family. For large discrete action spaces (LLMs): policy gradient or DPO-family.

## Common pitfalls

- **Skipping the target network.** Loss explodes; training diverges.
- **Skipping experience replay.** Successive samples are highly correlated; gradient estimates are biased.
- **Confusing on-policy and off-policy.** Q-learning is off-policy: you can learn from old data with a different policy.
- **Ignoring exploration.** Greedy policy from a randomly-initialized $Q$ is terrible; $\varepsilon$-greedy with decaying $\varepsilon$ is the standard.

## Related

- [Policy gradient](/reference/policy-gradient/). Alternative paradigm.
- [Markov chains](/reference/markov-chains/). MDP background.
