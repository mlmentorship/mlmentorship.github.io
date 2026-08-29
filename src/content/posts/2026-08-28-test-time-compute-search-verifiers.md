---
title: "Test-time compute, search, and verifiers"
description: "Spend extra inference compute on several candidate solutions, search, tools, or revision, then select results with evidence."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Test-time compute is extra work performed when answering a request rather than during model training. The system may sample several answers, search over steps, use tools, revise an answer, or run verifiers.

## Why AI labs care

A fixed model can often solve more tasks when it receives more inference compute. This creates a new control:

- small budget for easy requests;
- larger budget for hard or high-value requests;
- strict limits for latency-sensitive requests;
- verification before an answer is returned.

The system needs evidence that the extra compute improves the target result enough to justify its cost.

## Main methods

### Best-of-n sampling

Generate $n$ independent candidates and choose the highest verifier score.

This works when:

- the policy sometimes produces a correct answer;
- samples are diverse enough;
- the verifier ranks them well.

More samples give smaller gains when outputs are highly correlated.

### Self-consistency

Generate several reasoning paths and choose the answer that appears most often. This is useful when different valid paths reach the same answer.

Agreement can still be confidently wrong. It is not an external correctness check.

### Iterative revision

Generate an answer, critique it, and revise it. Revision helps when errors can be found from the available evidence. Repeated revision can also add confident errors or unnecessary length.

### Search

Treat partial solutions as states. Expand promising states, score them, and keep a limited set.

Examples include beam-like search, tree search, and planning with tools. Search quality depends on the proposal policy, state representation, score, and stopping rule.

### Tool use

Use a calculator, code runner, search index, theorem checker, or external environment. Tool results can provide stronger evidence than internal confidence.

## The verifier is the bottleneck

A strong policy with a weak verifier may generate a correct answer and then reject it. A weak policy with an exploitable verifier may learn to produce high-scoring failures.

A verifier may be:

- an exact program;
- protected unit tests;
- a symbolic solver;
- an environment state check;
- a model judge;
- a human reviewer.

Use direct checks before model judgment. Test the verifier on hard negatives and unusual valid solutions.

## Allocate compute by request

Uniformly spending a large budget is wasteful. A deployment system should estimate:

- task difficulty;
- value of a correct answer;
- cost of a wrong answer;
- expected gain from more samples or search;
- remaining latency and token budget;
- verifier confidence.

Stop when the expected value of more work is below its cost, or when a hard budget is reached.

A simple policy can use a small initial sample. Escalate only when candidates disagree, verification fails, or the task belongs to a high-risk class.

## Small example: coding assistant

A coding assistant receives a repository task.

1. Produce one patch and run focused tests.
2. If tests fail, use the failure output to revise.
3. If tests pass, run protected regressions and static checks.
4. For a high-risk change, generate a second plan or patch for comparison.
5. Stop when checks pass and the patch stays within scope.
6. Escalate when tests are missing or requirements remain unclear.

Generating ten patches before running any test wastes compute and weakens feedback.

## Measure the compute-quality curve

For each budget, report:

- task success;
- verifier false accepts and false rejects;
- tokens and tool calls;
- latency percentiles;
- monetary or hardware cost;
- failure rate by task difficulty;
- gains relative to a stronger base model.

Compare systems at equal total cost as well as equal sample count.

## Main failure modes

### Correlated candidates

Many samples repeat one approach. Increase diversity through prompts, temperatures, policies, or search state rather than only increasing $n$.

### Verifier exploitation

The search finds outputs that score well without solving the task.

### Search error

A weak early score prunes the path that would have led to the best answer.

### Unbounded loops

Revision or tool use continues without enough progress. Set step, token, time, and cost limits.

### Latency tails

Average latency looks acceptable while difficult requests exceed the service target. Report percentiles and overload behavior.

### Hidden quality loss

Shortcuts reduce cost or latency while harming an important slice. Keep held-out evaluations and production guardrails.

## In an interview

Use this order:

1. Define the task, risk, and budget.
2. Choose sampling, revision, search, or tools.
3. Define the verifier and test its errors.
4. Explain how compute is allocated and stopped.
5. Measure a quality-cost-latency curve.
6. Cover correlated samples, verifier exploits, and tail latency.
7. Compare against training or serving a stronger base model.

## Common mistakes

- Saying "sample more" without a selection method.
- Treating majority agreement as correctness.
- Using a model judge without calibration.
- Ignoring candidate correlation.
- Reporting accuracy without tokens, latency, and cost.
- Allowing open-ended agent loops.
- Comparing systems at different total compute without saying so.

*Related: [train and serve a reasoning model under a fixed budget](/questions/design-reasoning-model-fixed-budget/), [annotated reasoning-strategy mock](/guides/annotated-reasoning-strategy-mock/), [evaluate an agent](/questions/evaluate-an-agent/), and [RL with verifiable rewards and GRPO](/concepts/rl-verifiable-rewards-grpo/).*