---
title: "Design post-training data, an RL environment, and its grader"
description: "Turn one capability into episodes, evidence-bearing graders, adversarial data, and a training signal that cannot reward the wrong process."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Improve a tool-using agent's ability to resolve support tickets. Design the data, environment, grader, training loop, and evaluation. The agent must not access unauthorized account data.

Begin with behavior, not an RL algorithm. Define what a successful trajectory looks like, which actions are forbidden, and what evidence distinguishes genuine success from reward hacking. PPO, DPO, or another optimizer cannot rescue an environment that rewards the wrong thing.

## The design sequence

### 1. Define the capability distribution

Specify ticket types, user states, tools, permissions, ambiguity, long-horizon depth, and failure severity. A single average success rate will hide whether the model improves on easy password resets while failing high-risk billing changes.

### 2. Define episode semantics

An episode needs:

- observable state and hidden ground truth;
- available actions and tool schemas;
- permission boundaries;
- stochasticity and reset behavior;
- terminal success, terminal failure, and timeout;
- an audit log sufficient to reproduce the trajectory.

### 3. Separate outcome from process

Use a structured grade rather than one opaque scalar:

- **task success:** was the ticket actually resolved?
- **policy compliance:** were permissions and required confirmations respected?
- **process quality:** were actions efficient, grounded, and non-duplicative?
- **communication quality:** was the user-facing result accurate and calibrated?

Some violations are gates, not tradeable penalties. Unauthorized access cannot be offset by a polite final response.

### 4. Choose the grader stack

Use deterministic checks for permissions, tool arguments, final database state, and known invariants. Use model graders for qualities that require semantic judgment, but calibrate them against blinded human labels and adversarial examples. Reserve human review for ambiguous, high-impact, or novel failures.

Every grade should include evidence: the action, rule, comparison, or rubric item that produced it.

### 5. Construct training data

Combine:

- expert trajectories for difficult or safety-critical behavior;
- model-generated trajectories filtered by verifiers;
- contrastive pairs that isolate one decision;
- failed trajectories with labeled failure points;
- adversarial cases that target grader blind spots;
- held-out families that cannot leak into training.

Data diversity is not prompt paraphrase count. Vary state, tools, incentives, ambiguity, and consequences.

## What an L4 answer sounds like

> "Collect good conversations, use an LLM judge to score success, and train with RL. Give positive reward for resolution and negative reward for unsafe actions."

The answer names the pipeline but leaves the attack surface undefined. The model judge is treated as truth, safety is a small penalty, and there is no environment version, evidence, or held-out generalization.

## What an L5 answer adds

An L5 candidate specifies episode state and actions, separates deterministic and model-based grading, makes hard policy violations disqualifying, and defines coverage slices. They version environment, tools, grader, model, and data.

They also test the grader before training:

- swap a correct final answer onto an unsafe trajectory;
- fabricate tool output while preserving the outcome;
- repeat no-op actions;
- exploit formatting or verbosity preferences;
- find a valid but unusual solution;
- withhold information the grader assumes is present.

The grader should rank behavior for the intended reasons.

## What an L6 answer adds

An L6 candidate treats the environment as a research instrument. They measure inter-rater agreement, grader calibration, exploitability, coverage, variance, and sensitivity to model scale. They know that training against a fixed judge changes the distribution of attacks on that judge.

They separate three evaluations:

1. **capability:** can the model solve the task under representative conditions?
2. **alignment or policy:** does it preserve constraints under pressure and adversarial context?
3. **product value:** does the intervention improve real outcomes without unacceptable latency, cost, or user harm?

They define a launch decision and a rollback signal. Benchmark movement alone does not ship a model.

## Tells that get you a strong-hire vote

- Capability and prohibited behavior are explicit.
- Episode reset, timeout, and terminal conditions are defined.
- Safety-critical violations are gates, not small penalties.
- Graders return evidence and are calibrated against humans.
- Training and held-out families are separated by behavior, not wording.
- Adversarial episodes target reward hacking.
- Environment and grader versions accompany every result.
- Online product evidence remains separate from training reward.

## Tells that get you down-leveled

- Starting with PPO or DPO.
- One model judge and one scalar reward.
- Rewarding the final answer while ignoring the trajectory.
- Treating synthetic volume as data quality.
- No held-out behavioral families.
- No grader false-positive analysis.
- Shipping because the reward curve increased.

## Common follow-up

"The policy learns to call the verifier repeatedly until one noisy score passes. What changes?"

The environment should expose repeated calls and charge or cap them. The grader should evaluate the trajectory, not only the best sampled score. More importantly, investigate why the verifier is noisy enough to exploit and whether training data now overrepresents that loophole. A penalty can suppress the symptom; a calibrated verifier and explicit process contract address the cause.

Try the [post-training environment lab](/prep/labs/post-training-environment/) before reading this as a checklist.

*Related: [preference data and reward models](/concepts/preference-data-and-reward-models/), [RL environments and graders](/concepts/rl-environments-and-graders/), and [evaluate an agent](/questions/evaluate-an-agent/).*
