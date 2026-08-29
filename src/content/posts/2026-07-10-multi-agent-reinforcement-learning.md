---
title: "Multi-agent reinforcement learning"
description: "Learning when other agents change the environment: non-stationarity, credit assignment, coordination, competition, and evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Single-agent RL assumes a stationary environment. The moment other learning agents share it, that assumption breaks: markets, game-playing, traffic, negotiation, ad auctions, and self-play training are all multi-agent. Multi-agent RL studies environments where several agents act at once, and each agent's reward and transitions depend on the other agents' policies, which are themselves changing.

## Why it is harder than single-agent RL

- **Non-stationarity:** from one agent's view, the environment shifts as the others learn.
- **Credit assignment:** a shared team reward does not say whose action helped.
- **Partial observability:** agents usually see only local information.
- **Coordination equilibria:** several stable conventions can coexist.
- **Opponent modeling:** competitive agents adapt strategically to you.
- **Evaluation:** doing well against one set of opponents need not generalize.

## Centralized training, decentralized execution

The common pattern trains critics or value functions with global state and all agents' actions, while each deployed policy acts on its local observation only. MADDPG and value-decomposition methods apply this idea differently.

## Cooperative and competitive methods

**Cooperative.** Value decomposition builds a team value from per-agent values: VDN sums them; QMIX uses a monotonic mixing network so decentralized greedy actions stay consistent with the joint value.

**Competitive and mixed.** Self-play, population-based training, opponent sampling, and league systems reduce overfitting to a single opponent. Nash equilibrium is a useful reference point but hard to compute in large stochastic games.

## In an interview

1. Name the source of non-stationarity.
2. Clarify cooperative, competitive, or mixed incentives.
3. Explain centralized training with decentralized execution.
4. Address credit assignment and communication.
5. Evaluate against diverse policies, unseen partners, and exploiters.

## Common confusions

- **"Just treat other agents as part of the environment."** Their policies change and respond strategically.
- **"Self-play guarantees robustness."** It can cycle or overfit to its own population.
- **"A team reward creates teamwork."** It can also create free-riding and muddy credit assignment.

*Related: [actor-critic methods](/concepts/actor-critic-methods/), [exploration versus exploitation](/concepts/exploration-vs-exploitation/), and [PPO](/concepts/ppo/).*
