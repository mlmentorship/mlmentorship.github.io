---
title: "Contextual bandits"
description: "Choose actions from context while balancing reward and uncertainty. The bridge between supervised prediction, experimentation, and reinforcement learning."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Why it matters

Bandits show up everywhere a system chooses an action and only learns about the action it took: recommendation, notifications, ranking, treatment selection, adaptive experiments. They formalize the cost that logged production data hides: you know what happened under the policy you ran, not what would have happened under the alternatives you never tried.

Formally, a contextual bandit sees context $x_t$, chooses action $a_t$, and receives a reward only for that action, maximizing cumulative reward while learning which actions work for which contexts. It sits between two neighbors: unlike full reinforcement learning, the action does not drive a persistent state transition; unlike supervised learning, the labels for the actions you did not choose are missing by design.

## Core approaches

- **Epsilon-greedy:** exploit most of the time; pick randomly with probability $\epsilon$.
- **UCB:** choose the highest estimated reward plus an uncertainty bonus.
- **Thompson sampling:** sample parameters from the posterior and act greedily under that sample.
- **LinUCB / linear Thompson sampling:** assume expected reward is linear in the context features.

## Off-policy evaluation

Logged data needs propensities. Inverse propensity scoring estimates a target policy by weighting reward by the probability of the logged action; doubly robust estimators combine a reward model with that correction. Without exploration support, a policy that chooses actions absent from the log is not identifiable offline: you have no evidence about what they would have returned.

## In an interview

1. Separate contextual bandits from supervised learning and MDPs.
2. Define regret and the exploration-exploitation trade-off.
3. Describe UCB or Thompson sampling.
4. Explain logging propensities and off-policy evaluation.
5. Cover delayed rewards, non-stationarity, safety constraints, and feedback loops.

## Common confusions

- **"An A/B test is a bandit."** A fixed A/B test explores with a static policy; a bandit adapts assignment over time.
- **"Bandits always beat experiments."** Adaptive policies complicate inference and can chase short-term proxies.
- **"Use historical clicks as labels."** Only the actions the logging policy chose have outcomes; selection bias is the whole problem.

*Related: [exploration versus exploitation](/concepts/exploration-vs-exploitation/), [A/B testing for ML](/concepts/ab-testing-for-ml/), and [policy gradient](/concepts/policy-gradient/).*
