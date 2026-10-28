---
title: "Value-based vs. policy-based RL"
description: "Trace values, action selectors, actors and critics; choose data reuse and learned dynamics independently of the action interface."
date: "2026-04-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A value function scores actions; an actor produces actions or distributions. Choose data reuse and dynamics models separately.

## Problem

For one state, actions are left, stay and right.
Their return estimates are $Q=[1.2,0.4,2.1]$.
An explicit actor instead gives probabilities $[0.15,0.10,0.75]$.
Trace each action path, then add epsilon-greedy behavior with $\epsilon=0.2$ and uniform exploration across all three actions.

## Picture: Scores Need a Selection Rule

<!-- visual:value-policy-action-interface -->
<figure class="learning-figure" aria-labelledby="value-policy-interface-title">
	<p class="visual-kicker">One state, two action interfaces</p>
	<p class="visual-title" id="value-policy-interface-title">What learned object sits directly upstream of the action?</p>
	<div class="visual-grid--two" role="group" aria-label="Value-based control derives an action through a selector over learned action values; policy-based control obtains an action directly from a learned policy; actor-critic uses an actor for action choice and a critic for a training signal">
		<section class="visual-panel" aria-labelledby="value-interface-title">
			<h4 id="value-interface-title">Value: scores then a rule</h4>
			<p>Left: 1.2. Stay: 0.4. Right: 2.1.</p>
			<p>Greedy selection chooses right. Epsilon-greedy instead chooses right with probability 13/15.</p>
			<p><code>state &rarr; Q scores &rarr; rule &rarr; action</code></p>
			<p>Bellman targets can train the values; the behavior rule need not be deterministic.</p>
		</section>
		<section class="visual-panel" aria-labelledby="policy-interface-title">
			<h4 id="policy-interface-title">Policy: learn the selector</h4>
			<p>Left: 0.15. Stay: 0.10. Right: 0.75.</p>
			<p>Sampling chooses right 75% of the time; using the mode always chooses right here.</p>
			<p><code>state &rarr; actor &rarr; action</code></p>
			<p>An actor can also be deterministic; a stochastic actor can use a return-weighted log-probability gradient.</p>
		</section>
		<section class="visual-panel" style="grid-column: 1 / -1;" aria-labelledby="actor-critic-interface-title">
			<h4 id="actor-critic-interface-title">Actor-critic: separate training path</h4>
			<p><code>transition &rarr; critic &rarr; baseline/target/actor derivative</code></p>
			<p>A state/action critic supports learning; it need not choose deployed actions.</p>
			<p>SAC combines a stochastic actor with off-policy Q learning, not an actor-equals-on-policy rule.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> follow the action paths before the training path. Values plus a rule and a learned actor are different interfaces, not opposite definitions of randomness. <span class="visual-reader-check" data-reader-check="value-policy-action-interface"><strong>Check:</strong> Why can a value-based agent act stochastically? What declared rule gives right probability 13/15?</span></figcaption>
</figure>

## Check the Behavior Probabilities

For epsilon-greedy here, right receives the greedy branch plus its share of exploration:

$$
P(\text{right}\mid s)=0.8+0.2/3=13/15.
$$

Left and stay each receive $1/15$.
The actor's right probability remains $0.75$, despite both interfaces having the same mode.
State behavior/target distributions and corrections explicitly when reusing data.

## Change 1: Discrete Choices Become Continuous Torque

**New task.** Choose real-valued motor commands.

**Picture and change.** Price continuous Q maximization or use a bounded actor with controller/safety constraints.

**Check.** A small finite output head cannot enumerate every torque.

**Use when.** Argmax is no longer a cheap scan.

## Change 2: Interaction Becomes Expensive

**New task.** Real-world samples are costly and the simulator is imperfect.

**Picture and change.** Compare reuse and learned-model planning with support, model-error, transfer and safe-fallback checks.

**Check.** Dynamics errors can compound; sample efficiency is not guaranteed.

**Use when.** Data cost changes the feasible methods.

## Failure: An Algorithm Name Replaces Its Assumptions

A task-name table can hide unsupported offline actions, unsafe controls or missing observation memory.
No universal variance or sample-efficiency ranking resolves those constraints.

## Transfer Check

Is model-based learning mutually exclusive with actor-critic?

### Reasoned Answer

No. Learned dynamics can support planning or training data alongside an actor and critic.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Value-based methods

Q-learning/DQN-family methods learn action values with chosen behavior and target rules.
Replay, target networks and double estimators address specific data/stability risks, not unsupported offline actions automatically.

## Policy-based methods

Actors may be stochastic or deterministic.
PPO commonly reuses recent rollouts for several updates; SAC uses replay under its algorithm.
Monte Carlo gradients can have high variance; baselines and critics introduce their own estimation considerations.

## Actor-critic: combining the two

A critic can provide a state baseline, bootstrapped target or action derivative.
SAC is stochastic/off-policy; TD3 uses a deterministic actor with Q critics.

## Decision matrix

| Assumption | Candidate direction | Discriminating check |
| --- | --- | --- |
| Modest finite actions | Enumerated Q values or actor | Selection/coverage cost |
| Continuous control | Actor or continuous optimizer | Bounds, stability, latency |
| Offline/costly data | Supported reuse or models | Coverage, shift, safe interaction |
| Partial observation | Memory/belief with either interface | Missing information and history |
| Preference fitting | Specified preference objective | DPO is not generic sequential online RL |

## What about model-based RL?

Dynamics learning/planning is another axis, not an exclusive third interface.
Dreamer/MuZero-style designs still need model-error and decision checks.

## Common pitfalls

Do not equate actors with fresh-only data, values with determinism, or critics with guaranteed improvement.

## Related

- [Q-learning](/concepts/q-learning/). Canonical value-based.
- [Policy gradient](/concepts/policy-gradient/). Canonical policy-based.
- [PPO](/concepts/ppo/). Modern actor-critic.

## Sources

- [Sutton et al., 1999](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html).
- [SAC documentation](https://spinningup.openai.com/en/latest/algorithms/sac.html).

</div>
