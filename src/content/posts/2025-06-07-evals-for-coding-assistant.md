---
title: "How would you build evals for a coding assistant?"
description: "Code is one of the few LLM domains where ground truth is verifiable. Use that. The senior answer combines verifiable metrics with human review for what verification can't catch."
date: "2025-06-07"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team interviews at companies building coding products.*

The L4 candidate names HumanEval. The L6 candidate distinguishes the metrics that matter for a *product* from the leaderboard scores, and addresses what verification can and can't catch.

## What an L4 answer sounds like

> "Use HumanEval and MBPP. Run the generated code and check if it passes the tests."

This is leaderboard eval, useful for vendor selection but useless for a product. You've consumed benchmarks but haven't built one.

## What an L5 answer sounds like

> "Code eval has a special property: ground truth is *executable*. Lean into it.
>
> Three layers:
>
> 1. **Functional correctness via tests.** For each task: a problem statement, a reference solution, and a test suite. Generated code passes if the tests pass. Public benchmarks (HumanEval, MBPP, SWE-bench, LiveCodeBench) provide this format. For your own product, build an internal benchmark from real user requests with hand-written tests.
>
> 2. **Code quality.** Tests passing isn't enough; the code should also be readable, idiomatic, and not introduce bugs in adjacent code. Linters and static analysis catch some of this. LLM judges catch the rest, but need calibration.
>
> 3. **Real-task evaluation.** For an IDE assistant: completion acceptance rate, edit-distance from generated to final code, time-to-task-completion. These are the online metrics that actually predict business value.
>
> Public benchmarks are useful for vendor selection. Real-task metrics are what you ship on."

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
> **Hallucinated APIs are the dominant failure mode.** Model invents methods that don't exist. Standard verification catches them; without execution-based eval they slip through.
>
> **For online metrics**: completion acceptance rate is the standard, but it's noisy (users accept things they later modify). Time-to-final-code (until the user stops editing) is a better signal."

## Tells that get you a strong-hire vote

- You distinguish **benchmarks (vendor selection)** from **product metrics (shipping)**.
- You bring up **test contamination** as a benchmark gotcha.
- You mention **SWE-bench** as a more realistic evaluator.
- You name **hallucinated APIs** as the dominant failure mode.
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
