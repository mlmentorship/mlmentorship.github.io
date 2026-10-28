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

**Learning objective:** given a proposed safety technique, identify whether it improves judgment under a capability gap, limits harm under intentional subversion, or contributes evidence to both.

<!-- visual:oversight-control-assumption-test -->
<figure class="learning-figure editorial-illustration" aria-labelledby="oversight-control-assumption-test-edition-title"><span class="visual-fragment-anchor" id="oversight-control-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="oversight-control-shared-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="oversight-path-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="control-path-title" aria-hidden="true"></span><p class="visual-title" id="oversight-control-assumption-test-edition-title">Checking a hard answer and containing an intentional attacker need different evidence</p><svg viewBox="0 0 360 529" role="img" aria-labelledby="oversight-control-assumption-test-edition-svg-title oversight-control-assumption-test-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="oversight-control-assumption-test-edition-svg-title">Checking a hard answer and containing an intentional attacker need different evidence</title><desc id="oversight-control-assumption-test-edition-svg-desc">The same fictional difficult code task is evaluated under two distinct scenarios. Without assuming attack, assisted decomposition exposes a boundary-case counterexample and tests evaluator correctness. With intentional untrusted action assumed, a trusted scoped boundary denies an unapproved action and records unchanged protected state, testing containment rather than answer quality.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional hard code task; two separate</tspan><tspan x="180" dy="19">threat assumptions</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">OVERSIGHT</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">CONTROL</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Assumption</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="92" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="157" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">hard to judge</tspan><tspan x="90" dy="19">attack not</tspan><tspan x="90" dy="19">required</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="92" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="147.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">intentional</tspan><tspan x="270" dy="19">attacker</tspan><tspan x="270" dy="19">untrusted</tspan><tspan x="270" dy="19">actions</tspan></text></g><path d="M15 232L345 232" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="252" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Concrete evidence</tspan></text><g data-editorial-node><rect x="15" y="266" width="150" height="92" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="288.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">assisted edge</tspan><tspan x="90" dy="19">test</tspan><tspan x="90" dy="19">finds wrong</tspan><tspan x="90" dy="19">answer</tspan></text></g><g data-editorial-node><rect x="195" y="266" width="150" height="92" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="288.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">unapproved</tspan><tspan x="270" dy="19">action denied</tspan><tspan x="270" dy="19">protected state</tspan><tspan x="270" dy="19">unchanged</tspan></text></g><path d="M15 373L345 373" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="393" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Success test</tspan></text><g data-editorial-node><rect x="15" y="407" width="150" height="92" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="439" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">distinguish</tspan><tspan x="90" dy="19">correct</tspan><tspan x="90" dy="19">from faulty work</tspan></text></g><g data-editorial-node><rect x="195" y="407" width="150" height="92" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="429.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">contain</tspan><tspan x="270" dy="19">attempted action</tspan><tspan x="270" dy="19">under trusted</tspan><tspan x="270" dy="19">boundary</tspan></text></g></svg><figcaption><strong>Read it this way:</strong> Assistance tests whether the evaluator can judge difficult work. Containment tests a trusted boundary under the intentional-attacker assumption. Neither observation substitutes for the other. <span class="visual-reader-check" data-reader-check="oversight-control-assumption-test"><strong>Check:</strong> Which observation tests evaluator competence? Which tests containment under the explicitly adversarial assumption?</span></figcaption></figure>

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
