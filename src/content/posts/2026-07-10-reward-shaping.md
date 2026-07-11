---
title: "Reward shaping"
description: "Modify learning signals without accidentally changing the task, creating reward hacking, or hiding specification failure."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Why it matters

Sparse or delayed rewards (robotics, games, long-horizon agents, recommender objectives) make credit assignment hard: the agent rarely sees a signal, so it rarely learns. Reward shaping adds auxiliary feedback to densify that signal. The danger is that changing the reward can change the optimal policy, and a badly shaped reward produces confident, well-optimized behavior that does the wrong thing: circling near a waypoint, farming easy interactions, or maximizing proxy engagement while destroying long-term value.

A shaped reward has the form

$$r'(s,a,s') = r(s,a,s') + F(s,a,s').$$

## Potential-based shaping

The safe construction makes $F$ a difference of a potential function $\Phi$:

$$F(s,a,s') = \gamma \Phi(s') - \Phi(s).$$

This shifts value estimates while preserving the set of optimal policies under standard assumptions. It rewards progress toward useful states without redefining the final objective, which is why it is the default when you must shape at all.

## Design procedure

1. Write down the true objective and the behavior you will not accept.
2. Identify why credit assignment is hard.
3. Prefer state potentials or demonstrations over arbitrary event bonuses.
4. Check whether a policy can maximize the shaped reward without doing the task.
5. Evaluate on the original reward and independent guardrails.
6. Anneal or remove the shaping once it is no longer needed.

## In an interview

Explain sparse credit assignment, potential-based shaping, reward hacking, and how you would red-team the proxy. The senior move is to separate optimization failure from specification failure: a perfectly optimized bad reward is not an algorithm bug, it is a spec bug.

## Common confusions

- **"More detailed reward is always better."** More terms mean more loopholes and unstable scales.
- **"Human preference solves specification."** Preference data still has annotator, coverage, and manipulation limits.
- **"The training reward is the evaluation."** Evaluate against independent task outcomes and safety constraints.

*Related: [policy gradient](/concepts/policy-gradient/), [PPO](/concepts/ppo/), and [RLHF and DPO](/concepts/rlhf-and-dpo/).*
