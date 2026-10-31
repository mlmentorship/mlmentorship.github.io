---
title: "How do you evaluate an agent?"
description: "Agent eval is harder than chat eval because there are intermediate steps, tool calls, and long-horizon outcomes. The senior answer evaluates trajectories, not just final outputs."
date: "2025-11-11"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team and agent-platform interviews.*

Evaluate whether a tool-using agent completes authorized tasks reliably across entire trajectories, not just individual generated answers.

## Problem

**Context.** A hypothetical agent plans multiple steps, calls tools, observes results, and may create persistent external effects.
A candidate model or orchestration change must be assessed before wider release.

**Task.** Define task-level success, trajectory and step checks, an environment and task suite, repeated trials, and a staged deployment decision.
Explain how offline results will be compared with observed production behavior.

**Constraints.** Separate answer quality, tool correctness, authorization, and external-state outcomes.
Control environment versions and budgets; account for stochasticity, retries, long-horizon failures, and consequential actions without executing unsafe tests.

## Why agent eval is harder

A chat model produces one output per input. An agent produces a *trajectory*: plan, tool calls, intermediate results, final answer. Each step can fail, and failures compound. Standard end-to-end eval can't tell you whether the planner was wrong, the tool returned bad data, or the synthesis was bad.

<!-- visual:agent-eval-diagnostic-flywheel -->
<figure class="learning-figure editorial-illustration" aria-labelledby="agent-eval-diagnostic-flywheel-edition-title"><p class="visual-title" id="agent-eval-diagnostic-flywheel-edition-title">A passing test and an unauthorized write can occur in the same trial</p><div class="visual-argument" data-visual-argument="agent-eval-diagnostic-flywheel" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 112" role="img" aria-label="A passed test and an unauthorized write can coexist. One trial can complete its task while violating the authority boundary. The combined result must fail the declared policy gate."><title>A passed test and an unauthorized write can coexist</title><desc>One trial can complete its task while violating the authority boundary. The combined result must fail the declared policy gate.</desc><text x="112.25" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="112.25" dy="0">task test</tspan></text><text x="203.35" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="203.35" dy="0">authorized</tspan></text><text x="294.45000000000005" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="294.45000000000005" dy="0">joint</tspan></text><text x="54.69999999999999" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="54.69999999999999" dy="0">trial</tspan></text><g data-proof-status="present"><rect data-proof-cell="0,0" x="66.69999999999999" y="36" width="86.10000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M102.75 54l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="109.75" y="82" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="109.75" dy="0">pass</tspan></text><path d="M71.69999999999999 89L147.8 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="absent"><rect data-proof-cell="0,1" x="157.8" y="36" width="86.10000000000001" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="200.85000000000002" cy="54" r="9" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M195.85000000000002 54L205.85000000000002 54" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="200.85000000000002" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="200.85000000000002" dy="0">NO</tspan></text><path data-proof-warning="0,1" d="M161.8 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="stopped"><rect data-proof-cell="0,2" x="248.9" y="36" width="86.10000000000001" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><path d="M283.95 46l16 16m0-16l-16 16" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5"></path><text x="291.95" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="291.95" dy="0">FAIL</tspan></text><path data-proof-warning="0,2" d="M252.9 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 454" role="img" aria-labelledby="agent-eval-diagnostic-flywheel-edition-svg-title agent-eval-diagnostic-flywheel-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="agent-eval-diagnostic-flywheel-edition-svg-title">A passing test step and a failing task outcome can coexist</title><desc id="agent-eval-diagnostic-flywheel-edition-svg-desc">A fictional coding-agent trial edits the intended file but then writes an unauthorized file. The intended test passes, the task safety outcome fails, and the trial uses four tool calls. The new unauthorized-write slice returns to the fixed-budget suite; a production mismatch reopens its coverage assumptions.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional trial T4; same trajectory, three</tspan><tspan x="180" dy="19">lenses</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">WHAT HAPPENED</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">WHAT IT PROVES</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Outcome</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="154" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">intended test</tspan><tspan x="90" dy="19">passes</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="154" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">not sufficient</tspan><tspan x="270" dy="19">for success</tspan></text></g><path d="M15 207L345 207" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="227" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Failure step</tspan></text><g data-editorial-node><rect x="15" y="241" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="90" y="270" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">writes forbidden</tspan><tspan x="90" dy="19">file</tspan></text></g><g data-editorial-node><rect x="195" y="241" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="270" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">safety outcome</tspan><tspan x="270" dy="19">FAIL</tspan></text></g><path d="M15 323L345 323" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="343" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Resource ledger</tspan></text><g data-editorial-node><rect x="15" y="357" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="395.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">4 tool calls</tspan></text></g><g data-editorial-node><rect x="195" y="357" width="150" height="67" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="386" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">compare at fixed</tspan><tspan x="270" dy="19">budget</tspan></text></g></svg><div class="visual-derivation-caption">Add the unauthorized-write slice to the suite. A production mismatch reopens coverage. Passing one test does not erase the same trial's safety failure.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="agent-eval-diagnostic-flywheel"><strong>Check:</strong> How can two trials have the same final outcome but different trajectory failures or costs?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="agent-eval-diagnostic-flywheel">One trial can complete its task while violating the authority boundary. The combined result must fail the declared policy gate.</span></figcaption></figure>

## What an L4 answer sounds like

> "I'd run the agent on a benchmark of tasks and measure final-answer accuracy. Compare against a baseline model."

This treats the agent as a black box and gives you one number with no diagnostic value. You've used agents but haven't debugged one.

## What an L5 answer sounds like

> "Agent eval needs three levels:
>
> 1. **Final-task success.** Did the agent complete the task? Binary or rubric-graded. The headline number.
>
> 2. **Trajectory-level metrics.** Step count, tool-call accuracy, plan adherence, recovery from intermediate failures. Surfaces where the agent is *failing* (planning vs tool use vs synthesis).
>
> 3. **Step-level eval.** For each tool call: was it the right tool? Were the arguments correct? For each plan step: did it move toward the goal? Often graded by an LLM judge against a rubric.
>
> Build a golden set of ~100-300 tasks stratified by difficulty and category. Evaluate at all three levels per release. Track the offline-online correlation: does success on the eval set predict success in production?"

This is L5. You've broken the eval into the right levels and named the operational concern.

## What an L6 answer adds

> "...practical things:
>
> **Cost is a first-class metric for agents.** A 90% successful agent that takes 50 tool calls per task is often worse than an 85% successful agent that takes 5. Report cost per task, success at fixed cost, or Pareto curves.
>
> **Failure-mode taxonomy is the most useful artifact.** Categorize failures: planning errors, tool-call errors, hallucinated tool results, infinite loops, premature termination, refusal-when-should-have-tried. Track frequencies over releases. The taxonomy itself takes weeks to build but pays back forever.
>
> **Multi-turn evals are necessary for conversational agents.** Single-turn benchmarks miss issues that only appear under conversation drift. Build multi-turn scenarios with realistic user replies (often LLM-simulated, validated against real conversations).
>
> **Trajectory replay** is the agent equivalent of the 'read the outputs' practice in LLM eval. Watch the agent solve 10 tasks per week, end-to-end. Patterns you'll only see by watching: agents giving up too early, repeating the same failed action, asking for clarification when they shouldn't.
>
> **Production-online vs offline gap.** Real users phrase tasks differently than your eval set assumes. Sample 50 production traces per week, score them against your rubric, compare to your offline trends. If they diverge, the offline eval is overfit to its assumptions."

## Tells that get you a strong-hire vote

- You evaluate at **multiple granularities**: task, trajectory, step.
- You bring up **cost per task** and **Pareto curves**.
- You build a **failure-mode taxonomy**.
- You discuss **trajectory replay** as a debugging practice.
- You track the **offline-online correlation** explicitly.

## Tells that get you down-leveled

- "Run a benchmark and report accuracy."
- No step-level eval.
- No cost metric.
- Treating multi-turn as the same problem as single-turn.

## Common follow-up

"How do you handle stochasticity in agent eval?"

The L6 answer:

> "First define the sampling unit and the claim. Repeated attempts estimate variation on the fixed suite; independently sampled tasks support a different generalization claim. Suggested counts such as 5-10 attempts are starting points, not universally sufficient release criteria.
>
> Compare candidate and baseline on matched tasks under the same budgets, environment versions, and outcome rubric. State whether success means one attempt, eventual success after retries, or success within a total budget. Count failed attempts and retries in cost.
>
> A binomial interval needs an appropriate independent-trial model. Repeated heterogeneous tasks are not automatically independent observations for a new-task claim. When tasks are the sampled units, preserve that clustering in uncertainty estimates, for example by resampling paired task results rather than pooling every attempt.
>
> Temperature and seeds can reduce some sampling variation; they do not guarantee deterministic hardware, tools, or hosted execution. Report the task-level differences and uncertainty, not a single lucky run. The [application-evaluation lesson](/questions/how-would-you-evaluate-an-llm-application/) supplies the separate outcome, support, and release checks."

**Predict before revealing:** two tasks each run ten times produce twenty outcomes. Have you evaluated twenty independently sampled tasks?

<details><summary>Keep attempts and task clusters separate</summary>

No. There are twenty attempts nested within two task clusters.
Those outcomes can characterize repeatability on the fixed pair, but new-task uncertainty cannot be justified by pretending there were twenty task draws.
State how tasks were sampled, preserve candidate/baseline pairing, and gate consequential authorization failures separately from average success.

</details>

---

*Related: [enterprise agent-platform design](/questions/design-enterprise-agent-platform/), [LLM Evals essay](/guides/llm-evals-the-hardest-part/), [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), and [How do you handle hallucinations in production?](/questions/handle-hallucinations-in-production/).*
