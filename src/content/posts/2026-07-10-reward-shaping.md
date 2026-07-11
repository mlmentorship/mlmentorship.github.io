---
title: "Reward shaping"
description: "Modify learning signals without accidentally changing the task, creating reward hacking, or hiding specification failure."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Definition

Reward shaping adds auxiliary feedback to make sparse or delayed reinforcement-learning problems easier to learn. A shaped reward often has the form

$$r'(s,a,s') = r(s,a,s') + F(s,a,s').$$

The danger is that changing reward can change the optimal policy.

## Potential-based shaping

A policy-invariant form uses a potential function $\Phi$:

$$F(s,a,s') = \gamma \Phi(s') - \Phi(s).$$

This shifts value estimates while preserving optimal policies under standard assumptions. It rewards progress toward useful states without redefining the final objective.

## Why it matters

Sparse-reward robotics, games, agents, and recommender objectives often need denser learning signals. Poor shaping creates shortcuts: circling near a waypoint, farming easy interactions, or optimizing proxy engagement while harming long-term value.

## Design procedure

1. Write the true objective and unacceptable behavior.
2. Identify why credit assignment is difficult.
3. Prefer state potentials or demonstrations over arbitrary event bonuses.
4. Test whether a policy can maximize the shaped reward without accomplishing the task.
5. Evaluate on the original reward and independent guardrails.
6. Anneal or remove shaping when possible.

## Interview answer

Explain sparse credit assignment, potential-based shaping, reward hacking, and how you would red-team the proxy. Senior answers distinguish optimization failure from specification failure: a perfectly optimized bad reward is not an RL algorithm bug.

## Common confusions

- **“More detailed reward is always better.”** More terms create more loopholes and unstable scales.
- **“Human preference solves specification.”** Preference data has annotator, coverage, and manipulation limits.
- **“The training reward is the evaluation.”** Evaluate against independent task outcomes and safety constraints.

*Related: [policy gradient](/concepts/policy-gradient/), [PPO](/concepts/ppo/), and [RLHF and DPO](/concepts/rlhf-and-dpo/).*
