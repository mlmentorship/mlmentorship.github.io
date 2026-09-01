---
title: "Q-learning"
description: "Learn the action-value function Q(s, a) by Bellman backups. The foundation of value-based RL. DQN, Rainbow, and the original Atari breakthroughs."
date: "2026-02-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Q-learning** [(Watkins, 1989)](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) is an off-policy temporal-difference algorithm that learns the optimal action-value function $Q^*(s, a)$. The expected return from taking action $a$ in state $s$ and then acting optimally. By iterating the Bellman optimality update:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big].
$$

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

**Learning objective:** given one transition and the next state's action values, distinguish the action sampled by the behavior policy from the greedy action used to construct the Q-learning target.

<!-- visual:q-learning-off-policy-backup -->
<figure class="learning-figure plot-panel" aria-labelledby="q-learning-backup-title">
	<p class="visual-kicker">One off-policy backup</p>
	<p class="visual-title" id="q-learning-backup-title">The trajectory can explore while the target looks ahead greedily.</p>
	<svg viewBox="0 0 360 440" role="img" aria-labelledby="q-learning-backup-svg-title q-learning-backup-svg-desc">
		<title id="q-learning-backup-svg-title">Q-learning separates behavior from the greedy backup target</title>
		<desc id="q-learning-backup-svg-desc">A sampled transition moves from state s with action a and current value 4 to next state s prime with reward 2. At s prime, three possible next actions have Q-values: up 2, right 6, and down 1. The behavior policy samples down to keep exploring, but that sampled next action is not used in this update. The target policy selects the maximum value 6 from right. With discount one half, the target is 2 plus one half times 6, which equals 5. The temporal-difference error is 5 minus 4, or 1. With learning rate one quarter, the updated current Q-value is 4.25.</desc>
		<defs>
			<marker id="q-learning-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-edge)"></path></marker>
		</defs>
		<text class="viz-axis-label" x="18" y="22">1 · OBSERVE ONE TRANSITION</text>
		<rect class="viz-node viz-node--input" x="24" y="38" width="126" height="62" rx="4"></rect>
		<text class="viz-callout" x="87" y="61" text-anchor="middle">current pair (s, a)</text>
		<text class="viz-label" x="87" y="81" text-anchor="middle">Q(s, a) = 4</text>
		<path d="M150 69H204" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#q-learning-arrow)"></path>
		<text class="viz-label" x="177" y="57" text-anchor="middle">reward r = 2</text>
		<rect class="viz-node viz-node--state" x="205" y="38" width="131" height="62" rx="4"></rect>
		<text class="viz-callout" x="271" y="61" text-anchor="middle">next state s′</text>
		<text class="viz-label" x="271" y="81" text-anchor="middle">consider every action</text>
		<text class="viz-axis-label" x="18" y="128">2 · SEPARATE BEHAVIOR FROM TARGET</text>
		<rect class="viz-node" x="24" y="145" width="312" height="126" rx="4"></rect>
		<rect class="viz-node viz-node--focus" x="38" y="211" width="280" height="28" rx="3"></rect>
		<text class="viz-callout" x="43" y="168">possible action a′</text>
		<text class="viz-callout" x="248" y="168">Q(s′, a′)</text>
		<path class="viz-gridline" d="M42 179H318 M42 207H318 M42 235H318"></path>
		<text class="viz-label" x="43" y="198">↑ up</text>
		<text class="viz-label" x="43" y="226">→ right</text>
		<text class="viz-label" x="43" y="254">↓ down</text>
		<text class="viz-callout" x="266" y="198" text-anchor="middle">2</text>
		<text class="viz-callout" x="266" y="226" text-anchor="middle">6 · MAX</text>
		<text class="viz-callout" x="266" y="254" text-anchor="middle">1</text>
		<path d="M90 271V297" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4"></path>
		<text class="viz-gradient-label" x="180" y="289">behavior samples down; not used in this target</text>
		<text class="viz-axis-label" x="18" y="318">3 · MOVE THE CURRENT VALUE TOWARD THE TARGET</text>
		<rect class="viz-node viz-node--focus" x="24" y="335" width="312" height="45" rx="4"></rect>
		<text class="viz-callout" x="180" y="354" text-anchor="middle">target = r + γ max Q(s′, a′) = 2 + 0.5 × 6 = 5</text>
		<text class="viz-label" x="180" y="371" text-anchor="middle">TD error = target − current = 5 − 4 = 1</text>
		<path d="M180 380V399" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#q-learning-arrow)"></path>
		<rect class="viz-node viz-node--output" x="68" y="400" width="224" height="32" rx="4"></rect>
		<text class="viz-callout" x="180" y="421" text-anchor="middle">Q(s, a) ← 4 + 0.25 × 1 = 4.25</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> use the sampled transition to obtain $s'$ and $r$, then scan every action value at $s'$ for the maximum. Here exploration samples down, but the target uses right because its value is 6. That mismatch is intentional: behavior gathers data, while the greedy target defines what Q-learning learns.</figcaption>
</figure>

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

- [Policy gradient](/concepts/policy-gradient/). Alternative paradigm.
- [Markov chains](/concepts/markov-chains/). MDP background.
