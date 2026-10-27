---
title: "Markov decision processes and Bellman equations"
description: "Define sequential decisions with states, actions, transitions, rewards, and value functions before choosing an RL algorithm."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A Markov decision process specifies sequential decisions through sufficient state, available actions, transitions, rewards, and task boundaries.
Bellman equations combine immediate reward with continuation value, then average the environment's possible outcomes.

In our hypothetical support task, asking for one fact either resolves the request or needs follow-up.
Compute this fixed action's expected return before comparing it with another action.

## MDP components

An MDP is often written as $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$: states;
- $\mathcal{A}(s)$: actions available in state $s$;
- $P(s'\mid s,a)$: chance of next state $s'$ after action $a$ in state $s$;
- $R(s,a,s')$: reward for the transition;
- $\gamma$: discount factor, with $0\leq\gamma<1$ for bounded-reward infinite-horizon guarantees.

A policy $\pi(a\mid s)$ gives a distribution over actions in each state.

The Markov property requires state to retain the information predicting transitions and rewards, given the chosen action.
Permission checks constrain actions before reward optimization.

## Return and value

The discounted return from time $t$ is:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots.
$$

The state value under policy $\pi$ is:

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t=s].
$$

The action value is:

$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a].
$$

$V$ averages the policy's current action; $Q$ fixes that action, then follows the policy afterward.
Here $r_t$ is the reward from the transition beginning at time $t$.

## Bellman expectation equations

A value can be written one step at a time:

$$
V^\pi(s) = \mathbb{E}_{a\sim\pi,\,s'\sim P}
\left[R(s,a,s') + \gamma V^\pi(s')\right].
$$

For action values:

$$
Q^\pi(s,a) = \mathbb{E}_{s'\sim P}
\left[R(s,a,s') + \gamma\mathbb{E}_{a'\sim\pi}Q^\pi(s',a')\right].
$$

For a terminal indicator $D$, the target is $R+\gamma(1-D)V^\pi(s')$.
Actual termination contributes no continuation, even if a stored terminal value is nonzero.

<!-- visual:bellman-stochastic-backup -->
<figure class="learning-figure plot-panel" aria-labelledby="bellman-backup-title">
	<p class="visual-kicker">One Bellman backup</p>
	<p class="visual-title" id="bellman-backup-title">Expectation combines every possible next-state return.</p>
	<p><strong>Action gate:</strong> choose only authorized actions before evaluating rewards; reward cannot grant permission.</p>
	<svg viewBox="0 0 360 400" role="img" aria-labelledby="bellman-backup-svg-title bellman-backup-svg-desc">
		<title id="bellman-backup-svg-title">A stochastic action-value Bellman backup</title>
		<desc id="bellman-backup-svg-desc">From the current support state, the agent takes the action ask for one missing fact. The environment then resolves the request with probability three quarters, giving immediate reward 2 and terminal continuation value 0, or needs follow-up with probability one quarter, giving immediate reward negative 1 and next-state value 4. With discount one half, the two branch returns are 2 and 1. Their probability-weighted average is Q pi of s comma a equals 1.75.</desc>
		<defs>
			<marker id="bellman-backup-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-edge)"></path></marker>
		</defs>
		<text class="viz-axis-label" x="18" y="22">1 · FIX THE CURRENT STATE AND ACTION</text>
		<rect class="viz-node viz-node--input" x="27" y="38" width="130" height="55" rx="4"></rect>
		<text class="viz-callout" x="92" y="61" text-anchor="middle">state s</text>
		<text class="viz-label" x="92" y="79" text-anchor="middle">missing one fact</text>
		<path d="M157 65H201" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#bellman-backup-arrow)"></path>
		<text class="viz-label" x="179" y="54" text-anchor="middle">action a</text>
		<rect class="viz-node viz-node--focus" x="202" y="38" width="131" height="55" rx="4"></rect>
		<text class="viz-callout" x="268" y="61" text-anchor="middle">ask for fact</text>
		<text class="viz-label" x="268" y="79" text-anchor="middle">environment responds</text>
		<text class="viz-axis-label" x="18" y="112">2 · SCORE EACH POSSIBLE NEXT STATE</text>
		<path d="M268 94V120L104 151M268 120L256 151" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#bellman-backup-arrow)"></path>
		<text class="viz-callout" x="123" y="140">P = 0.75</text>
		<text class="viz-callout" x="260" y="142">P = 0.25</text>
		<rect class="viz-node viz-node--output" x="27" y="153" width="147" height="91" rx="4"></rect>
		<text class="viz-callout" x="101" y="176" text-anchor="middle">resolved (terminal)</text>
		<text class="viz-label" x="101" y="198" text-anchor="middle">reward r = +2</text>
		<text class="viz-label" x="101" y="216" text-anchor="middle">next value V(s′) = 0</text>
		<text class="viz-callout" x="101" y="237" text-anchor="middle">r + γV = 2</text>
		<rect class="viz-node viz-node--state" x="186" y="153" width="147" height="91" rx="4"></rect>
		<text class="viz-callout" x="260" y="176" text-anchor="middle">needs follow-up</text>
		<text class="viz-label" x="260" y="198" text-anchor="middle">reward r = −1</text>
		<text class="viz-label" x="260" y="216" text-anchor="middle">next value V(s′) = 4</text>
		<text class="viz-callout" x="260" y="237" text-anchor="middle">r + γV = 1</text>
		<text class="viz-label" x="180" y="268" text-anchor="middle">discount γ = 0.5 on both continuation values</text>
		<path d="M101 245V292L180 316M260 245V292L180 316" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#bellman-backup-arrow)"></path>
		<text class="viz-axis-label" x="18" y="301">3 · WEIGHT THE BRANCH RETURNS</text>
		<rect class="viz-node viz-node--focus" x="27" y="320" width="306" height="60" rx="4"></rect>
		<text class="viz-callout" x="180" y="344" text-anchor="middle">Qᵖⁱ(s,a) = 0.75(2) + 0.25(1)</text>
		<text class="viz-callout" x="180" y="367" text-anchor="middle">= 1.75</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> first fix an available action. The resolved branch returns 2; follow-up returns −1 + 0.5 × 4 = 1. Weight those returns by 0.75 and 0.25, giving 1.75. An unweighted average gives 1.5; choosing the best random branch gives 2. Neither is this expectation. Original schematic checked against <a href="https://incompleteideas.net/book/the-book-2nd.html">Sutton and Barto (2018)</a>.</figcaption>
</figure>

## Bellman optimality

The optimal action value satisfies:

$$
Q^*(s,a) = \mathbb{E}_{s'\sim P}
\left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right].
$$

The maximum chooses a future available action, not the most favorable stochastic next state.
For bounded rewards and $\gamma<1$, the Bellman operator is a contraction in the maximum norm.

### Evaluate a different policy

**New task:** evaluate the current follow-up policy rather than its best possible action.

**Picture and change:** follow-up offers terminal action returns $0$ and $8$.
The current policy chooses them with probabilities $0.75$ and $0.25$, so continuation value becomes $2$.

**Check:** the original action now has value $0.75(2)+0.25(-1+0.5\times2)=1.5$.
Always choosing the follow-up return $8$ instead gives $2.25$.
Those answer different policy questions.

**Use when:** the estimand is a declared policy's return, not optimal continuation.

## Small example: a support agent

**New task:** two identical messages arrive after different verification histories.

**Picture and change:** retain trusted verification state $v$ and enforce the available-action set.
When $v=0$, allow asking or escalating; when $v=1$, a policy-compliant refund may also be available.

**Check:** assigning refund reward $1000$ cannot enable that action when $v=0$.
The visible message alone is insufficient state; inferred confidence cannot manufacture authorization.

**Use when:** omitted history changes predictions or valid actions.

<!-- learning:core-end -->

<div data-learning-extension>

## Why AI labs care

Control, scheduling, recommendations, language-model post-training, and tool-using agents all require a declared decision problem.
An algorithm name cannot replace its transition, information, or authority contract.

## Partial observability

Noisy observations may omit the Markov state.
A belief distribution or justified memory summarizes relevant history; a text prompt is not automatically sufficient.
Trusted permission state remains authoritative even when a policy maintains a belief about hidden conditions.

## Horizon and discount

Finite-horizon values need time-to-go, either in state or in time-indexed policies and values.
An undiscounted continuing task needs additional convergence assumptions; $\gamma=1$ alone supplies no finite-value guarantee.
Discounting changes reward preferences, not hard task boundaries.
Termination and external truncation need separate bootstrap policies.

## In an interview

Can the same location require different actions with one step or two steps remaining?

<details><summary>Answer</summary>

Yes. Cashing out gives $1$ immediately; waiting gives zero now and enables terminal reward $2$ next step.
With $\gamma=1$, cash out when only one step remains; wait when two remain.
Location alone cannot encode that distinction for a stationary policy.

</details>

## Common mistakes

An omitted verification bit plus a callable forbidden action creates a defective environment contract.
Penalties do not repair that access path.
State estimation, policy evaluation, and authorization answer different questions.
Q-learning samples an optimality target; policy-gradient methods optimize policy parameters using their own estimators.

</div>

*Related: [Q-learning](/concepts/q-learning/), [policy-gradient methods](/concepts/policy-gradient/), and [RL environments and graders](/concepts/rl-environments-and-graders/).*