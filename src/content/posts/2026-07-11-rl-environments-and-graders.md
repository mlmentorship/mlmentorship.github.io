---
title: "RL environments and graders for language-model agents"
description: "The environment defines what behavior is possible; the grader defines what optimization values. Both need versioning, adversarial tests, and evidence."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

An agent RL environment defines observations, actions, transitions, tools, and terminal conditions; a grader converts the resulting trajectory into structured evidence and a training signal.

## Why it matters

For language-model agents, the environment is part simulator, part benchmark, and part specification. If it omits a permission boundary, rewards only the final answer, or leaks the expected solution, training can improve the score while degrading the intended behavior.

The grader is not a passive metric. It is an objective the policy will search for exploits.

## Environment contract

Define:

- initial-state distribution;
- observable and hidden state;
- tool schemas and side effects;
- identity, permissions, and resource budgets;
- stochastic transition behavior;
- action validation;
- success, failure, timeout, and reset;
- reproducibility and event log;
- versioned dependencies and external services.

A support agent and a coding agent need different truth. One may validate database state and permission checks; the other needs repository state, tests, sandbox behavior, and patch scope.

## Grader stack

### Deterministic graders

Use executable checks for exact state, tests, policy rules, budgets, or verifier outputs. They are reproducible but cover only formalized behavior.

### Model graders

Use models for semantic correctness, relevance, communication, or open-ended quality. They scale but have bias, variance, prompt sensitivity, and attack surfaces.

### Human review

Use humans for ambiguous, novel, or high-consequence cases and to calibrate automated graders. Human labels also vary and need a protocol.

A strong system combines them and returns component evidence rather than one unexplained reward.

## Outcome and process

Separate:

- final task result;
- policy compliance;
- trajectory efficiency;
- tool grounding;
- communication;
- uncertainty or abstention;
- side effects.

Some constraints are hard gates. A correct answer produced through unauthorized data access is not partially successful.

## Coverage

Build episodes across task difficulty, tool failures, ambiguous instructions, adversarial context, long horizon, sparse reward, permission changes, and unusual but valid solutions. Hold out behavioral families to test generalization.

Avoid contamination. If training sees exact evaluator tasks or grader artifacts, the benchmark measures recall and exploit learning rather than capability transfer.

## Grader evaluation

Before training, test:

- human agreement and calibration;
- false positives on unusual valid behavior;
- false negatives on polished unsafe behavior;
- invariance to irrelevant style and length;
- sensitivity to the intended behavior;
- adversarial optimization;
- stability across model and environment versions;
- evidence quality for debugging.

After training, repeat with on-policy trajectories. The policy changes the grader's input distribution.

## Common confusions

- **"An environment is a prompt and expected answer."** Agent environments include state, actions, transitions, side effects, and termination.
- **"One reward is simpler."** It hides which behavior improved and lets components compensate incorrectly.
- **"Unit tests prove coding-agent quality."** Tests can be incomplete, weakened, or overfit.
- **"A model grader understands intent."** It predicts from a prompt and can share target-model blind spots.
- **"More episodes mean better coverage."** Near-duplicate tasks inflate count without expanding behavior.
- **"Offline reward improvement means product improvement."** Real users, tools, and incentives can differ.

## In an interview

Start with the capability and forbidden behavior. Then define episode semantics, grader evidence, data distribution, adversarial cases, contamination controls, versioning, and the external decision the reward supports.

*Related: [preference data and reward models](/concepts/preference-data-and-reward-models/), [reward shaping](/concepts/reward-shaping/), and [post-training environment lab](/prep/labs/post-training-environment/).*
