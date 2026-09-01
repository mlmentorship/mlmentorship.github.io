---
title: "Value-based vs. policy-based RL"
description: "Two paradigms in reinforcement learning. Value-based learns Q(s, a) and acts greedily; policy-based directly parametrizes the policy. When to use which."
date: "2026-04-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Value-based** RL learns a value function (typically $Q(s, a)$) and acts greedily with respect to it: $\pi(s) = \arg\max_a Q(s, a)$. **Policy-based** RL directly parametrizes a stochastic policy $\pi_\theta(a \mid s)$ and optimizes $\theta$ via the policy gradient. **Actor-critic** combines both.

Choosing the right paradigm is a central decision in RL system design. Mismatch leads to poor sample efficiency, instability, or simply not working. For instance, value-based methods are awkward in continuous action spaces, and pure policy-based methods are sample-inefficient in tabular settings.

**Learning objective:** trace how a state becomes an action in value-based and policy-based control, then locate the critic in an actor-critic without mistaking it for the action selector.

<!-- visual:value-policy-action-interface -->
<figure class="learning-figure" aria-labelledby="value-policy-interface-title">
	<p class="visual-kicker">Learning objective · one state, two action interfaces</p>
	<p class="visual-title" id="value-policy-interface-title">What learned object sits directly upstream of the action?</p>
	<div class="visual-grid--two" role="group" aria-label="Value-based control derives an action through a selector over learned action values; policy-based control obtains an action directly from a learned policy; actor-critic uses an actor for action choice and a critic for a training signal">
		<section class="visual-panel" aria-labelledby="value-interface-title">
			<h4 id="value-interface-title">VALUE-BASED · LEARN SCORES, THEN SELECT</h4>
			<p><strong>1 · Estimate</strong><br />For state <var>s</var>, learn one return estimate per candidate: <var>Q</var>(<var>s</var>, left) = 1.2, <var>Q</var>(<var>s</var>, stay) = 0.4, <var>Q</var>(<var>s</var>, right) = 2.1.</p>
			<p><strong>2 · Derive a policy</strong><br />An action selector reads those scores: greedy <code>argmax</code> chooses right; an exploration rule such as ε-greedy can sometimes choose another action.</p>
			<p><strong>Decision path</strong><br /><code>state → Q scores → selector → action</code></p>
			<p><strong>Canonical training signal</strong><br />A Bellman target moves the selected state-action value toward reward plus a bootstrapped future value.</p>
		</section>
		<section class="visual-panel" aria-labelledby="policy-interface-title">
			<h4 id="policy-interface-title">POLICY-BASED · LEARN THE SELECTOR ITSELF</h4>
			<p><strong>1 · Parameterize</strong><br />For the same state <var>s</var>, an explicit policy might output π(left | <var>s</var>) = 0.15, π(stay | <var>s</var>) = 0.10, π(right | <var>s</var>) = 0.75.</p>
			<p><strong>2 · Act directly</strong><br />Sample from that distribution, or use its mode at evaluation. For continuous control, the policy can instead output distribution parameters such as μ and σ.</p>
			<p><strong>Decision path</strong><br /><code>state → policy → action</code></p>
			<p><strong>Canonical training signal</strong><br />A return or advantage weights the gradient of the sampled action's log probability.</p>
		</section>
		<section class="visual-panel" style="grid-column: 1 / -1;" aria-labelledby="actor-critic-interface-title">
			<h4 id="actor-critic-interface-title">ACTOR-CRITIC · POLICY ACTION PATH, VALUE TRAINING PATH</h4>
			<p><strong>Decision:</strong> <code>state → actor πθ → action</code>. The actor remains the explicit policy that chooses the action.</p>
			<p><strong>Training:</strong> <code>transition → critic Vφ or Qφ → advantage / target → actor update</code>. The critic evaluates states or actions to reduce variance or construct an optimization target; it need not sit in the deployed action path.</p>
			<p><strong>Taxonomy check:</strong> actor-critic is a hybrid architecture, not a promise of on-policy learning. PPO is commonly on-policy; SAC is off-policy and reuses replay data.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> follow only the decision paths first. Value-based control must turn learned action scores into a policy through a selector; policy-based control learns that policy directly. Then read the actor-critic training path: the critic helps improve the actor, but the actor still supplies the action. These are distinctions about what is parameterized, not universal guarantees about stochasticity or data reuse. Original comparison informed by <a href="https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html">Sutton et al. (1999)</a> and <a href="https://spinningup.openai.com/en/latest/algorithms/sac.html">OpenAI Spinning Up's SAC documentation</a>.</figcaption>
</figure>

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

- [Q-learning](/concepts/q-learning/). Canonical value-based.
- [Policy gradient](/concepts/policy-gradient/). Canonical policy-based.
- [PPO](/concepts/ppo/). Modern actor-critic.
