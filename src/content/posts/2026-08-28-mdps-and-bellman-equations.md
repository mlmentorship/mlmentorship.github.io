---
title: "Markov decision processes and Bellman equations"
description: "Define sequential decisions with states, actions, transitions, rewards, and value functions before choosing an RL algorithm."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A Markov decision process, or MDP, models repeated decisions with states, actions, transition probabilities, rewards, and a time horizon. Bellman equations express a value as the current reward plus the value of the next state.

## Why AI labs care

MDPs are the base model for reinforcement learning. They appear in:

- game-playing agents;
- robot control;
- recommendation and ranking policies;
- language-model post-training;
- tool-using agents;
- resource scheduling.

An RL answer is hard to assess if the state, action, reward, and episode are not clear.

## MDP components

An MDP is often written as $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$: states;
- $\mathcal{A}$: actions;
- $P(s'\mid s,a)$: chance of next state $s'$ after action $a$ in state $s$;
- $R(s,a,s')$: reward for the transition;
- $\gamma$: discount factor between 0 and 1.

A policy $\pi(a\mid s)$ gives a distribution over actions in each state.

The Markov property says the current state contains the information needed to predict the next transition. If important history is missing, the state definition is incomplete.

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

$V$ asks how good a state is. $Q$ asks how good one action is in that state.

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

These equations are recursive. They turn a long-horizon problem into a one-step target plus a value estimate.

**Learning objective:** trace one action-value Bellman backup from a current state through stochastic next states, then probability-weight their immediate-reward-plus-discounted-value returns.

<!-- visual:bellman-stochastic-backup -->
<figure class="learning-figure plot-panel" aria-labelledby="bellman-backup-title">
	<p class="visual-kicker">One Bellman backup</p>
	<p class="visual-title" id="bellman-backup-title">Expectation combines every possible next-state return.</p>
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
	<figcaption><strong>Read it this way:</strong> fix the action first, then follow every transition the environment might produce. On each branch, add the immediate reward to the discounted next-state value; only then weight those branch returns by their transition probabilities. The backup is <em>not</em> the best-looking branch or an unweighted average: here it is <code>0.75 × 2 + 0.25 × 1 = 1.75</code>. Original schematic checked against <a href="https://incompleteideas.net/book/the-book-2nd.html">Sutton and Barto (2018)</a>.</figcaption>
</figure>

## Bellman optimality

The optimal action value satisfies:

$$
Q^*(s,a) = \mathbb{E}_{s'\sim P}
\left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right].
$$

Q-learning uses a sample version of this target. Policy-gradient methods optimize the policy directly instead of taking a max over learned action values.

## Small example: a support agent

A support agent can read an account, ask a question, issue a refund, or escalate.

- **State:** verified account facts, conversation history, policy state, and tool results.
- **Action:** one tool call or one message.
- **Transition:** customer reply or tool outcome.
- **Reward:** successful resolution, with hard penalties for unauthorized actions.
- **Terminal state:** resolved, escalated, refused, or timed out.

If the state omits whether identity was verified, the policy cannot make a safe refund decision. If reward measures only resolution, the policy may take unsafe shortcuts.

## Partial observability

Many real systems do not expose the full state. A robot has noisy sensors. A language-model agent does not know the user's hidden goal. This is a partially observable MDP.

The policy needs a belief or memory based on past observations. For an LLM agent, the prompt and tool history act as a limited memory. They may still omit important state.

## Horizon and discount

- A finite task may end after a fixed number of steps.
- An ongoing task may use discounting.
- A smaller $\gamma$ favors near rewards.
- A larger $\gamma$ gives more weight to long-term outcomes.

For bounded agent tasks, a clear terminal rule is often easier to reason about than choosing a discount factor without need.

## In an interview

Use this order:

1. Define state, action, transition, reward, and terminal conditions.
2. Check whether the state is Markov.
3. Define the policy and return.
4. Write the Bellman equation.
5. Explain whether value learning, policy learning, or planning fits the action space.
6. Discuss partial observability, reward failure, and offline versus online data.

## Common mistakes

- Calling the text prompt the whole environment.
- Leaving terminal conditions undefined.
- Using a reward that ignores safety or side effects.
- Confusing $V(s)$ with $Q(s,a)$.
- Using the optimality equation while evaluating a fixed policy.
- Treating a partially observed problem as fully observed.
- Choosing an RL algorithm before defining the MDP.

*Related: [Q-learning](/concepts/q-learning/), [policy-gradient methods](/concepts/policy-gradient/), and [RL environments and graders](/concepts/rl-environments-and-graders/).*