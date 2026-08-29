---
title: "Scalable oversight and AI control"
description: "Oversight tries to judge work beyond unaided human ability; control tries to keep systems safe even when the model may be strategically untrusted."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Scalable oversight** develops ways for humans and trusted systems to evaluate increasingly capable model work. **AI control** designs protocols that limit harm even if a powerful model may knowingly pursue an unwanted objective.

Both address a capability gap between the model and its overseer, but they make different assumptions.

Oversight often asks: how can a weaker evaluator reliably judge a stronger model's answer, plan, proof, or code?

Control asks a harsher question: if the model may be actively untrusted, what combination of monitoring, restricted actions, trusted models, audits, and containment keeps expected harm below a threshold?

A better judge is not automatically a control protocol. A sandbox is not automatically an evaluation method.

## Scalable oversight methods

### Decomposition

Break a task into subproblems a human or weaker model can judge. This works when local correctness composes into global correctness and the decomposition does not hide cross-cutting failures.

### Debate or critique

Models produce arguments, critiques, or competing answers so a judge can identify errors. The mechanism relies on truth or error being easier to verify when surfaced than to produce from scratch.

### Recursive assistance

Use model help to expand human evaluation capacity, such as summarizing evidence, generating tests, or tracing claims to sources. Assistance can also bias the evaluator or share the same blind spot.

### Process supervision

Evaluate intermediate steps rather than only outcomes. This can provide denser signal but requires that steps are meaningful, observable, and difficult to game.

### Verifiable environments

Use code execution, theorem checkers, simulators, or state transitions where correctness can be checked automatically. Verification covers only what the environment formalizes.

## Control protocols

A simple control system might combine:

- an untrusted high-capability model;
- a trusted but weaker monitor;
- restricted tools and least privilege;
- randomized audits;
- escalation to human review;
- canary tasks that reveal suspicious behavior;
- logging and containment;
- a policy that decides when to use, inspect, or block the untrusted model.

Evaluation should estimate usefulness and safety under an adaptive red team. A protocol that blocks everything is safe but useless. A protocol that succeeds only against non-adaptive attacks is fragile.

## Control evaluations

Define:

- threat model and untrusted-model capability;
- attack objective and budget;
- trusted components and their limitations;
- protocol and information available to each actor;
- safety failure and usefulness metrics;
- adaptive attack process;
- uncertainty and confidence in the estimated failure rate.

Rare catastrophic outcomes require careful statistical treatment. Zero observed failures in a small sample is not proof of negligible risk.

## Common confusions

- **"A stronger monitor solves oversight."** The target model can still exceed, manipulate, or share blind spots with the monitor.
- **"Decomposition always helps."** Some failures live in interactions between subproblems.
- **"Process supervision reveals intent."** Intermediate text can be incomplete or optimized for the supervisor.
- **"Sandboxing is sufficient control."** Tools, data, humans, and side channels define a larger boundary.
- **"A red team estimates real failure probability."** It demonstrates attacks under a search process; translating that to deployment risk requires reachability and prevalence assumptions.
- **"Safety and usefulness collapse to one score."** A protocol's trade-off frontier matters.

## In an interview

State which problem you are solving: evaluator capability, untrusted-model control, or both. Define actors, information, protocol, adaptive attack, safety metric, usefulness metric, and the evidence needed to trust the result.

*Related: [chain-of-thought monitorability](/concepts/chain-of-thought-monitorability/), [model organisms of misalignment](/concepts/model-organisms-of-misalignment/), and [design an LLM red-team program](/questions/design-llm-red-team-program/).*
