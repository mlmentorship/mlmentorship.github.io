---
title: "Multi-agent reinforcement learning"
description: "Learning when other agents change the environment: non-stationarity, credit assignment, coordination, competition, and evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Definition

Multi-agent RL studies environments where multiple learning agents act simultaneously. Each agent’s reward and transition distribution depend on other agents’ policies, which may also be changing.

## Why it is harder than single-agent RL

- **Non-stationarity:** from one agent’s view, the environment changes as others learn.
- **Credit assignment:** a team reward does not reveal which action helped.
- **Partial observability:** agents often see local information.
- **Coordination equilibria:** several stable conventions may exist.
- **Opponent modeling:** competitive agents adapt strategically.
- **Evaluation:** performance against one opponent set may not generalize.

## Centralized training, decentralized execution

A common pattern trains critics or value functions with global state and all agents’ actions, while each deployed policy uses only local observations. MADDPG and value-decomposition methods use this principle differently.

## Cooperative methods

Value decomposition represents a team value from per-agent values. VDN sums them; QMIX uses a monotonic mixing network so decentralized greedy actions remain consistent with the joint value.

## Competitive and mixed settings

Self-play, population-based training, opponent sampling, and league systems reduce overfitting to one policy. Nash equilibrium is a useful concept but difficult to compute in large stochastic games.

## Interview answer

1. Define the source of non-stationarity.
2. Clarify cooperative, competitive, or mixed incentives.
3. Explain centralized training and decentralized execution.
4. Address credit assignment and communication.
5. Evaluate against diverse policies, unseen partners, and exploiters.

## Common confusions

- **“Treat other agents as environment state.”** Their policies change and respond strategically.
- **“Self-play guarantees robustness.”** It can cycle or overfit to its own population.
- **“Team reward creates teamwork.”** It can create free-riding and poor credit assignment.

*Related: [actor–critic methods](/concepts/actor-critic-methods/), [exploration versus exploitation](/concepts/exploration-vs-exploitation/), and [PPO](/concepts/ppo/).*
