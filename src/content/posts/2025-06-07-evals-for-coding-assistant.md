---
title: "How would you build evals for a coding assistant?"
description: "Code is one of the few LLM domains where ground truth is verifiable. Use that. The senior answer combines verifiable metrics with human review for what verification can't catch."
date: "2025-06-07"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team interviews at companies building coding products.*

Evaluate a coding assistant's usefulness and risks using representative tasks rather than a single code-generation benchmark score.

## Problem

**Context.** A hypothetical coding product proposes completions, explains code, or edits repositories, and its model is changing.
You must decide whether a candidate version is better for developers.

**Task.** Design offline tasks, verification checks, product measurements, slice analysis, and a staged release decision.
Explain which failures unit tests, acceptance rates, and standard coding benchmarks cannot independently establish.

**Constraints.** Separate product modes, prevent evaluation leakage, and preserve private repository boundaries.
Measure correctness, latency, security, and downstream developer work; accepting a suggestion does not prove that it is correct.

## What an L4 answer sounds like

> "Use HumanEval and MBPP. Run the generated code and check if it passes the tests."

This is leaderboard eval, useful for vendor selection but useless for a product. You've consumed benchmarks but haven't built one.

## What an L5 answer sounds like

> "Code eval has a special property: ground truth is *executable*. Lean into it.
>
> Three layers:
>
> 1. **Functional checks via tests.** Bind the task, repository/environment version, and test oracle. Passing establishes the tested properties, not correctness outside that coverage. Public suites and internal user-derived tasks have different sampling and contamination risks; neither format alone makes the oracle complete.
>
> 2. **Code quality.** Use static analysis, a specific review rubric, and human-reviewed samples for properties outside the functional oracle. LLM judges can miss or invent defects; validate their agreement and failure modes instead of assuming they catch everything else.
>
> 3. **Real-task evaluation.** Acceptance, edits, and completion time are candidate product signals. Define the task endpoint, include review and repair, and use an appropriate controlled comparison before attributing productivity or business value to the assistant.
>
> Public benchmarks are useful for vendor selection. Real-task metrics are what you ship on."

**Learning objective:** choose the evaluation layer that can answer a coding-assistant claim, and state what evidence that layer still cannot provide.

<!-- visual:coding-assistant-evidence-ladder -->
<figure class="learning-figure editorial-illustration" aria-labelledby="coding-assistant-evidence-ladder-edition-title"><span class="visual-fragment-anchor" id="coding-eval-visual-title" aria-hidden="true"></span><p class="visual-title" id="coding-assistant-evidence-ladder-edition-title">Passing tests does not establish acceptance or sustained user value</p><div class="visual-argument" data-visual-argument="coding-assistant-evidence-ladder" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 112" role="img" aria-label="Passing tests does not establish sustained user value. A test receipt establishes only its tested contract. Acceptance, continued use, and durable value require separate observations."><title>Passing tests does not establish sustained user value</title><desc>A test receipt establishes only its tested contract. Acceptance, continued use, and durable value require separate observations.</desc><text x="123.28" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="123.28" dy="0">tests</tspan></text><text x="185.2" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="185.2" dy="0">accept</tspan></text><text x="247.12" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="247.12" dy="0">reuse</tspan></text><text x="309.03999999999996" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="309.03999999999996" dy="0">value</tspan></text><text x="80.32" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="80.32" dy="0">evidence</tspan></text><g data-proof-status="present"><rect data-proof-cell="0,0" x="92.32" y="36" width="56.92" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M113.78 54l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="120.78" y="82" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="120.78" dy="0">yes</tspan></text><path d="M97.32 89L144.24 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="unknown"><rect data-proof-cell="0,1" x="154.24" y="36" width="56.92" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="182.70000000000002" cy="54" r="10" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:2 3"></circle><text x="182.70000000000002" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="182.70000000000002" dy="0">?</tspan></text><path data-proof-warning="0,1" d="M158.24 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="unknown"><rect data-proof-cell="0,2" x="216.16" y="36" width="56.92" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="244.62" cy="54" r="10" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:2 3"></circle><text x="244.62" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="244.62" dy="0">?</tspan></text><path data-proof-warning="0,2" d="M220.16 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="unknown"><rect data-proof-cell="0,3" x="278.08" y="36" width="56.92" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="306.53999999999996" cy="54" r="10" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:2 3"></circle><text x="306.53999999999996" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="306.53999999999996" dy="0">?</tspan></text><path data-proof-warning="0,3" d="M282.08 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 472" role="img" aria-labelledby="coding-assistant-evidence-ladder-edition-svg-title coding-assistant-evidence-ladder-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="coding-assistant-evidence-ladder-edition-svg-title">The same patch supports three different kinds of claim</title><desc id="coding-assistant-evidence-ladder-edition-svg-desc">A fictional patch passes eight specified tests, then review finds a new dependency, and the developer accepts but rewrites the patch three hours later. Each observation supports only its own bounded claim. None alone establishes causal productivity improvement.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional patch P17; follow one artifact</tspan></text><text x="90" y="51" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">OBSERVATION</tspan></text><text x="270" y="51" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">SUPPORTED CLAIM</tspan></text><path d="M15 72L345 72" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="92" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Execute</tspan></text><g data-editorial-node><rect x="15" y="106" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="147.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">8 / 8 tests pass</tspan></text></g><g data-editorial-node><rect x="195" y="106" width="150" height="73" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="128.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">tested cases</tspan><tspan x="270" dy="19">pass</tspan><tspan x="270" dy="19">not all behavior</tspan></text></g><path d="M15 194L345 194" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="214" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Inspect</tspan></text><g data-editorial-node><rect x="15" y="228" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="260" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">new dependency</tspan><tspan x="90" dy="19">found</tspan></text></g><g data-editorial-node><rect x="195" y="228" width="150" height="73" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="250.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">maintenance</tspan><tspan x="270" dy="19">concern</tspan><tspan x="270" dy="19">needs review</tspan></text></g><path d="M15 316L345 316" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="336" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Observe</tspan></text><g data-editorial-node><rect x="15" y="350" width="150" height="92" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="382" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">accepted -&gt;</tspan><tspan x="90" dy="19">rewritten</tspan><tspan x="90" dy="19">after 3 hours</tspan></text></g><g data-editorial-node><rect x="195" y="350" width="150" height="92" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="270" y="372.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">acceptance is</tspan><tspan x="270" dy="19">not</tspan><tspan x="270" dy="19">lasting task</tspan><tspan x="270" dy="19">benefit</tspan></text></g></svg><div class="visual-derivation-caption">Follow P17 through all three observations. Tests, maintainability, and lasting usefulness are different claims. A causal benefit still needs a controlled comparison.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="coding-assistant-evidence-ladder"><strong>Check:</strong> Which user-value claim remains unsupported after the patch passes its tests?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="coding-assistant-evidence-ladder">A test receipt establishes only its tested contract. Acceptance, continued use, and durable value require separate observations.</span></figcaption></figure>

This is L5. You've named the three layers and distinguished benchmarks from product evals.

## What an L6 answer adds

> "...practical things:
>
> **Test contamination is rampant.** HumanEval, MBPP, and many other benchmarks are in training data. Inflated benchmark scores tell you nothing about generalization. Use newer or held-out benchmarks (LiveCodeBench rotates problems), or build internal benchmarks from non-public sources.
>
> **Multi-file, multi-step tasks are where benchmarks fall behind.** SWE-bench (full repo task completion) is closer to real product use than HumanEval. The gap between HumanEval scores and SWE-bench scores is substantial: a model can be at the top of HumanEval and useless on real codebases.
>
> **Tool use matters.** Modern coding agents use search, file read/write, test execution, debugger. Eval should test the *agent*, not just the model: can it find the right files, write the right tests, run them, iterate?
>
> **Per-language slicing matters.** A model can be excellent at Python, mediocre at Java, terrible at Rust. Aggregate scores hide this. Slice by language at minimum; by codebase pattern (web vs systems vs ML) for maturity.
>
> **Inventory failure modes rather than assuming a winner.** Invented APIs are one class; incorrect requirements, incomplete edits, regressions, authorization errors, and inadequate tests are others. Measure their frequencies on representative tasks. Execution and static checks have coverage limits.
>
> **For online metrics:** acceptance can precede later rewrites, and stopped editing can mean completion, abandonment, or interruption. Use an explicit verified task outcome and total developer effort rather than treating either proxy as a universal productivity measure. The [application-evaluation framework](/questions/how-would-you-evaluate-an-llm-application/) separates success evidence from release criteria."

## Tells that get you a strong-hire vote

- You distinguish **benchmarks (vendor selection)** from **product metrics (shipping)**.
- You bring up **test contamination** as a benchmark gotcha.
- You mention **SWE-bench** as a more realistic evaluator.
- You measure **invented APIs and other failure modes** instead of assuming their relative frequency.
- You discuss **slicing by language and codebase type**.

## Tells that get you down-leveled

- HumanEval as the main signal.
- No mention of contamination.
- No tool-use evaluation for agents.
- Treating "model" and "product" as interchangeable.

## Common follow-up

"How would you measure code quality beyond passing tests?"

The L6 answer:

> "Three signals: (1) static analysis (linter scores, complexity metrics, type-check pass), (2) LLM judges scoring code on a rubric (readability, idiomaticity, test coverage of generated code), validated against human judgments, and (3) human review on a sample. Code quality is fuzzier than correctness, so absolute scores matter less than tracking deltas across releases. A regression in lint warnings or in judge scores is a signal worth investigating even if functional correctness is unchanged."

---

*Related: [LLM Evals essay](/guides/llm-evals-the-hardest-part/), [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), [Evaluate an agent](/questions/evaluate-an-agent/).*
