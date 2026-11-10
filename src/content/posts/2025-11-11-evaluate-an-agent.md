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
<figure class="learning-figure editorial-illustration" aria-labelledby="agent-eval-diagnostic-flywheel-edition-title"><p class="visual-title" id="agent-eval-diagnostic-flywheel-edition-title">Evaluate the whole trajectory, verify external state, and count every trial and cost</p><div class="visual-argument" data-visual-argument="agent-eval-diagnostic-flywheel" data-argument-kind="matrix" data-argument-composition="trajectory-outcome-and-budget"><svg viewBox="0 0 360 1750" role="img" aria-label="Evaluate the whole trajectory, verify external state, and count every trial and cost. Reset versioned, contained environments and repeat representative tasks at fixed budgets. Inspect plans, tool arguments, receipts, recovery, and external state, not just final answers. This toy has two task completions but only one authorized success across three trials. Charge failed trials too. Diagnose the first failing step, repair its owner, and retest on held-out tasks. Calibrate graders, report uncertainty and safety separately, then compare offline evidence with staged production behavior."><title>Evaluate the whole trajectory, verify external state, and count every trial and cost</title><desc>Reset versioned, contained environments and repeat representative tasks at fixed budgets. Inspect plans, tool arguments, receipts, recovery, and external state, not just final answers. This toy has two task completions but only one authorized success across three trials. Charge failed trials too. Diagnose the first failing step, repair its owner, and retest on held-out tasks. Calibrate graders, report uncertainty and safety separately, then compare offline evidence with staged production behavior.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Inspect action evidence and joint</tspan><tspan x="20" dy="18">outcomes at fixed budgets</tspan></text><text x="20" y="110" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Toy: contained E3, tools V2, reset</tspan><tspan x="20" dy="18">each trial; ten-call budget. Account</tspan><tspan x="20" dy="18">A only.</tspan></text><text x="20" y="225" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Trial 2: task closes, but the</tspan><tspan x="20" dy="18">trajectory fails</tspan></text><circle cx="58" cy="340" r="26" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="58" y="346" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">OK</tspan></text><text x="110" y="332" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="110" dy="0">Read A: authorized</tspan></text><path d="M58 375L58 490M54 482L58 490L62 482" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="58" cy="530" r="26" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="58" y="536" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">X</tspan></text><text x="110" y="522" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="110" dy="0">Write B: unauthorized</tspan></text><path d="M58 565L58 680M54 672L58 680L62 672" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="58" cy="720" r="26" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="58" y="726" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">X</tspan></text><text x="110" y="712" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="110" dy="0">Receipt: synthetic</tspan><tspan x="110" dy="18">external effect</tspan></text><text x="20" y="820" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Task verifier: closed. Authority</tspan><tspan x="20" dy="18">verifier: fail. Inspect the first</tspan><tspan x="20" dy="18">violating call, not the final answer</tspan><tspan x="20" dy="18">alone.</tspan></text><text x="20" y="955" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Repeated reset trials: task /</tspan><tspan x="20" dy="18">authority / state</tspan></text><text x="20" y="1061" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">1</tspan></text><circle cx="80" cy="1055" r="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="80" y="1060" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="80" dy="0">OK</tspan></text><circle cx="150" cy="1055" r="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="150" y="1060" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="150" dy="0">OK</tspan></text><circle cx="220" cy="1055" r="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="220" y="1060" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="220" dy="0">OK</tspan></text><text x="270" y="1061" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="270" dy="0">4 calls</tspan></text><text x="20" y="1166" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">2</tspan></text><circle cx="80" cy="1160" r="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="80" y="1165" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="80" dy="0">OK</tspan></text><circle cx="150" cy="1160" r="20" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="150" y="1165" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="150" dy="0">X</tspan></text><circle cx="220" cy="1160" r="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="220" y="1165" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="220" dy="0">OK</tspan></text><text x="270" y="1166" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="270" dy="0">8 calls</tspan></text><text x="20" y="1271" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">3</tspan></text><circle cx="80" cy="1265" r="20" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="80" y="1270" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="80" dy="0">X</tspan></text><circle cx="150" cy="1265" r="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="150" y="1270" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="150" dy="0">OK</tspan></text><circle cx="220" cy="1265" r="20" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="220" y="1270" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="220" dy="0">X</tspan></text><text x="270" y="1271" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="270" dy="0">7 calls</tspan></text><text x="20" y="1400" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">2 of 3 tasks close; 1 of 3 jointly</tspan><tspan x="20" dy="18">pass. All trials cost 19 calls.</tspan></text><text x="20" y="1510" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Report uncertainty and success/cost</tspan><tspan x="20" dy="18">tradeoffs. These three toy trials are</tspan><tspan x="20" dy="18">not reliability certification.</tspan></text><text x="20" y="1620" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Diagnose, assign an owner, repair,</tspan><tspan x="20" dy="18">and retest on held-out tasks. Check</tspan><tspan x="20" dy="18">offline-to-production prediction</tspan><tspan x="20" dy="18">during staged rollout.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 454" role="img" aria-labelledby="agent-eval-diagnostic-flywheel-edition-svg-title agent-eval-diagnostic-flywheel-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="agent-eval-diagnostic-flywheel-edition-svg-title">A passing test step and a failing task outcome can coexist</title><desc id="agent-eval-diagnostic-flywheel-edition-svg-desc">A fictional coding-agent trial edits the intended file but then writes an unauthorized file. The intended test passes, the task safety outcome fails, and the trial uses four tool calls. The new unauthorized-write slice returns to the fixed-budget suite; a production mismatch reopens its coverage assumptions.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional trial T4; same trajectory, three</tspan><tspan x="180" dy="19">lenses</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">WHAT HAPPENED</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">WHAT IT PROVES</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Outcome</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="154" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">intended test</tspan><tspan x="90" dy="19">passes</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="154" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">not sufficient</tspan><tspan x="270" dy="19">for success</tspan></text></g><path d="M15 207L345 207" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="227" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Failure step</tspan></text><g data-editorial-node><rect x="15" y="241" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="90" y="270" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">writes forbidden</tspan><tspan x="90" dy="19">file</tspan></text></g><g data-editorial-node><rect x="195" y="241" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="270" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">safety outcome</tspan><tspan x="270" dy="19">FAIL</tspan></text></g><path d="M15 323L345 323" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="343" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Resource ledger</tspan></text><g data-editorial-node><rect x="15" y="357" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="395.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">4 tool calls</tspan></text></g><g data-editorial-node><rect x="195" y="357" width="150" height="67" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="386" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">compare at fixed</tspan><tspan x="270" dy="19">budget</tspan></text></g></svg><div class="visual-derivation-caption">Add the unauthorized-write slice to the suite. A production mismatch reopens coverage. Passing one test does not erase the same trial's safety failure.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="agent-eval-diagnostic-flywheel"><strong>Check:</strong> How can two trials have the same final outcome but different trajectory failures or costs?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="agent-eval-diagnostic-flywheel">Reset versioned, contained environments and repeat representative tasks at fixed budgets. Inspect plans, tool arguments, receipts, recovery, and external state, not just final answers. This toy has two task completions but only one authorized success across three trials. Charge failed trials too. Diagnose the first failing step, repair its owner, and retest on held-out tasks. Calibrate graders, report uncertainty and safety separately, then compare offline evidence with staged production behavior. <strong>Do two task completions mean two authorized successes?</strong></span></figcaption></figure>

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
