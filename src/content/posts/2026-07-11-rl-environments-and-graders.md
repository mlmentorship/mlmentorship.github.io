---
title: "RL environments and graders for language-model agents"
description: "The environment defines what behavior is possible; the grader defines what optimization values. Both need versioning, adversarial tests, and evidence."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An agent RL environment defines observations, actions, transitions, tools, and terminal conditions; a grader converts the resulting trajectory into structured evidence and a training signal.

For language-model agents, the environment is part simulator, part benchmark, and part specification. If it omits a permission boundary, rewards only the final answer, or leaks the expected solution, training can improve the score while degrading the intended behavior.

The grader is not a passive metric. It is an objective the policy will search for exploits.

**Learning objective:** trace one agent episode across the environment and grader contracts, then explain why a hard policy failure must remain separate from quality scores.

<!-- visual:environment-grader-contract-handoff -->
<figure class="learning-figure editorial-illustration" aria-labelledby="environment-grader-contract-handoff-edition-title"><span class="visual-fragment-anchor" id="environment-grader-handoff-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="environment-contract-lane-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="grader-contract-lane-title" aria-hidden="true"></span><p class="visual-title" id="environment-grader-contract-handoff-edition-title">A successful final state can still fail the declared process contract</p><div class="visual-argument" data-visual-argument="environment-grader-contract-handoff" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 112" role="img" aria-label="A successful final state can fail the required process. The environment reaches the goal while violating an allowed-tool constraint. The grader must retain both outcome and process evidence."><title>A successful final state can fail the required process</title><desc>The environment reaches the goal while violating an allowed-tool constraint. The grader must retain both outcome and process evidence.</desc><text x="105" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="105" dy="0">goal</tspan></text><text x="199" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="199" dy="0">process</tspan></text><text x="293" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="293" dy="0">trial</tspan></text><text x="46" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">run</tspan></text><g data-proof-status="present"><rect data-proof-cell="0,0" x="58" y="36" width="89" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M95.5 54l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="102.5" y="82" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="102.5" dy="0">yes</tspan></text><path d="M63 89L142 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="absent"><rect data-proof-cell="0,1" x="152" y="36" width="89" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="196.5" cy="54" r="9" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M191.5 54L201.5 54" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="196.5" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="196.5" dy="0">NO</tspan></text><path data-proof-warning="0,1" d="M156 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="stopped"><rect data-proof-cell="0,2" x="246" y="36" width="89" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><path d="M282.5 46l16 16m0-16l-16 16" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5"></path><text x="290.5" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="290.5" dy="0">FAIL</tspan></text><path data-proof-warning="0,2" d="M250 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 669" role="img" aria-labelledby="environment-grader-contract-handoff-edition-svg-title environment-grader-contract-handoff-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="environment-grader-contract-handoff-edition-svg-title">A successful final state can still fail the declared process contract</title><desc id="environment-grader-contract-handoff-edition-svg-desc">Fictional E17 closes owned order seventeen, then attempts a forbidden other-account read which is denied with no forbidden side effect. Immutable events reach a grader whose declared hard rule is zero unauthorized attempts. Task outcome passes but process fails, making the episode ineligible. Hidden grading evidence is not exposed to the policy.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional episode E17; declared</tspan><tspan x="180" dy="19">zero-unauthorized-attempt rule</tspan></text><text x="18" y="70" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">POLICY OBSERVES</tspan></text><g data-editorial-node><rect x="20" y="87" width="320" height="66" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="106" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Task: close own order 17</tspan><tspan x="180" dy="19">scoped identity; no hidden grader</tspan><tspan x="180" dy="19">answers</tspan></text></g><text x="18" y="197" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">IMMUTABLE EVENT EVIDENCE</tspan></text><g data-editorial-node><rect x="20" y="215" width="320" height="76" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="229.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">e1: own order 17 closed; allowed</tspan><tspan x="180" dy="19">e2: other-account read attempted;</tspan><tspan x="180" dy="19">DENIED</tspan><tspan x="180" dy="19">no forbidden side effect committed</tspan></text></g><path d="M180 291L180 326" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,319.85692206676737L180,326L183.35597877022943,319.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="18" y="350" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">GRADER ONLY: hidden target + actual final state</tspan></text><g data-editorial-node><rect x="15" y="386" width="150" height="100" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="90" y="422" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Task outcome</tspan><tspan x="90" dy="19">order 17 closed</tspan><tspan x="90" dy="19">SUCCESS</tspan></text></g><g data-editorial-node><rect x="195" y="386" width="150" height="100" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="403" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Process hard</tspan><tspan x="270" dy="19">gate</tspan><tspan x="270" dy="19">1 forbidden</tspan><tspan x="270" dy="19">attempt</tspan><tspan x="270" dy="19">FAIL</tspan></text></g><text x="180" y="535" style="font-size:13px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">INELIGIBLE under this declared grader</tspan><tspan x="180" dy="19">contract</tspan></text><text x="180" y="580" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Successful outcome cannot average away a hard</tspan><tspan x="180" dy="19">failure.</tspan></text><text x="180" y="620" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A different contract could grade blocked</tspan><tspan x="180" dy="19">attempts differently.</tspan></text></svg><div class="visual-derivation-caption">Read actual events, not just the final task state. This declared grader rejects even the blocked unauthorized attempt. Other contracts may differ; quality cannot override this hard rule.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="environment-grader-contract-handoff"><strong>Check:</strong> Which event makes the illustrated trajectory ineligible even if its final task score is high?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="environment-grader-contract-handoff">The environment reaches the goal while violating an allowed-tool constraint. The grader must retain both outcome and process evidence.</span></figcaption></figure>

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
