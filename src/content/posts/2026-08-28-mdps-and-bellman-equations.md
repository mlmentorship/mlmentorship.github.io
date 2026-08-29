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