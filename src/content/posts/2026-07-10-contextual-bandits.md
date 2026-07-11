---
title: "Contextual bandits"
description: "Choose actions from context while balancing reward and uncertainty. The bridge between supervised prediction, experimentation, and reinforcement learning."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Definition

A contextual bandit observes context $x_t$, chooses action $a_t$, and receives reward only for that action. The goal is to maximize cumulative reward while learning which actions work for which contexts.

Unlike full reinforcement learning, the action does not change a persistent state transition model. Unlike supervised learning, labels for unchosen actions are missing by design.

## Why it matters

Bandits appear in recommendation, notifications, ranking, treatment selection, and adaptive experiments. They formalize the exploration cost hidden by logged production data: the system knows what happened under the old policy, not what would have happened under alternatives.

## Core approaches

- **Epsilon-greedy:** exploit most of the time; choose randomly with probability $\epsilon$.
- **UCB:** choose high estimated reward plus an uncertainty bonus.
- **Thompson sampling:** sample model parameters from the posterior and act greedily under that sample.
- **LinUCB / linear Thompson sampling:** assume expected reward is linear in context features.

## Evaluation

Logged data requires propensities. Inverse propensity scoring estimates a target policy using reward weighted by the probability of the logged action. Doubly robust estimators combine a reward model with propensity correction.

Without exploration support, a new policy that chooses actions absent from the log is not identifiable offline.

## Interview answer

1. Distinguish contextual bandits from supervised learning and MDPs.
2. Define regret and the exploration–exploitation trade-off.
3. Describe UCB or Thompson sampling.
4. Explain logging propensities and off-policy evaluation.
5. Discuss delayed rewards, non-stationarity, safety constraints, and feedback loops.

## Common confusions

- **“A/B testing is a bandit.”** A fixed A/B test explores with a static policy; a bandit adapts assignment over time.
- **“Bandits always beat experiments.”** Adaptive policies complicate inference and can optimize short-term proxies.
- **“Use historical clicks as labels.”** Only actions chosen by the logging policy receive outcomes; selection bias matters.

*Related: [exploration versus exploitation](/concepts/exploration-vs-exploitation/), [A/B testing for ML](/concepts/ab-testing-for-ml/), and [policy gradient](/concepts/policy-gradient/).*
