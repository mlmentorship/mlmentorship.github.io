---
title: "Extend an unfamiliar ML codebase with an AI coding agent"
description: "The new coding signal is control: map the system, delegate bounded work, reject bad changes, and prove the result."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> You inherit a small ML evaluation service with failing tests. Fix a distributed metric bug, add a slice guardrail, and use the provided coding agent where it helps.

Repair a small ML evaluation service and demonstrate that its distributed metric and slice guardrail satisfy explicit correctness requirements.

## Problem

**Context.** In this interview exercise, an inherited evaluation codebase has failing tests and a provided coding agent is available.
The requested changes concern distributed metric aggregation and a slice-level release guardrail.

**Task.** Reproduce the failure, identify the execution path and invariant, implement the smallest repair, and add focused regression tests.
Explain any agent assistance, review its output, and present evidence that the requested behavior works.

**Constraints.** Follow the exercise's tool and data permissions; assistance is optional and does not transfer responsibility.
Preserve unrelated behavior, distinguish global from shard-level metrics, and do not weaken tests to make the suite pass.

## The first eight minutes

Write five lines before editing:

1. **Entry point:** where data enters.
2. **Execution path:** which functions transform it.
3. **Invariant:** what must remain true after the change.
4. **Verification:** the narrow test and the complete test command.
5. **Change surface:** the smallest set of files likely to move.

For the [agentic evaluation lab](/prep/labs/agentic-codebase/), the central invariant is mergeability:

$$
M(A \cup B) = M(A) \oplus M(B),
$$

where $M$ converts examples into sufficient statistics and $\oplus$ merges shard state. If this equality fails, a distributed evaluation can report a different score from a single-process run on identical data.

That statement gives the coding agent a target stronger than "fix the tests."

## Delegate outcomes, not ownership

A useful prompt has four parts:

- **Context:** the relevant file and call path.
- **Contract:** the invariant or acceptance criterion.
- **Boundary:** files and APIs that must not change.
- **Proof:** the test or measurement that demonstrates success.

<!-- visual:agent-review-control-loop -->
<figure class="learning-figure editorial-illustration" aria-labelledby="agent-review-control-loop-edition-title"><span class="visual-fragment-anchor" id="agent-control-title" aria-hidden="true"></span><p class="visual-title" id="agent-review-control-loop-edition-title">A passing agent test still misses the exact-expiry failure</p><div class="visual-argument" data-visual-argument="agent-review-control-loop" data-argument-kind="timeline" data-argument-composition="shared-clock"><svg viewBox="0 0 360 380" role="img" aria-label="The equality case distinguishes two nearly identical patches. Expiry equal to now must be rejected under this contract. A patch tested only before and after the boundary can miss it."><title>The equality case distinguishes two nearly identical patches</title><desc>Expiry equal to now must be rejected under this contract. A patch tested only before and after the boundary can miss it.</desc><path data-proof-boundary="0" d="M180 8V291" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.25;stroke-dasharray:2 5;opacity:.5"></path><text x="20" y="25" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">expiry &lt; now: reject</tspan></text><path d="M30 63L330 63" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="30" cy="63" r="7" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="53" y="94" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="53" dy="0">-1</tspan></text><path d="M180 50L180 79" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="122" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">expiry = now: must also reject</tspan></text><path d="M30 160L330 160" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="180" cy="160" r="7" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="180" y="191" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="180" dy="0">0</tspan></text><path d="M180 147L180 176" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="219" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">expiry &gt; now: allow</tspan></text><path d="M30 257L330 257" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="330" cy="257" r="7" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="307" y="288" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="307" dy="0">1</tspan></text><path d="M180 244L180 273" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M30 307L330 307" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M323 302L330 307" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M323 312L330 307" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="333" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="180" dy="0">expiry minus now; illustrative clock</tspan><tspan x="180" dy="18">units</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 473" role="img" aria-labelledby="agent-review-control-loop-edition-svg-title agent-review-control-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="agent-review-control-loop-edition-svg-title">A passing patch returns to an independent boundary test</title><desc id="agent-review-control-loop-edition-svg-desc">A fictional auth.py patch must deny expired tokens. Agent execution returns a diff and passing focused tests. The engineer independently tests expiry exactly equal to now, finds a failure, and returns the same bounded patch for revision. Acceptance remains outside the agent lane.</desc><text x="90" y="24" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">ENGINEER</tspan></text><text x="270" y="24" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">AGENT</tspan></text><path d="M180 43L180 405" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><g data-editorial-node><rect x="12" y="54" width="150" height="76" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="87" y="68.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="87" dy="0">Invariant:</tspan><tspan x="87" dy="19">expired token</tspan><tspan x="87" dy="19">denied</tspan><tspan x="87" dy="19">allowed: auth.py</tspan></text></g><path d="M162 92L198 92" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M191.8569220667674,95.35597877022943L198,92L191.8569220667674,88.64402122977057" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="198" y="54" width="150" height="76" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="273" y="78" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">Bounded patch</tspan><tspan x="273" dy="19">focused tests</tspan><tspan x="273" dy="19">actual diff</tspan></text></g><path d="M273 130L273 160" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M269.6440212297706,153.8569220667674L273,160L276.3559787702294,153.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="198" y="164" width="150" height="73" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="273" y="196" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">Return evidence</tspan><tspan x="273" dy="19">not acceptance</tspan></text></g><path d="M198 200L162 200" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M168.1430779332326,196.64402122977057L162,200L168.1430779332326,203.35597877022943" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="12" y="164" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="87" y="186.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="87" dy="0">Independent test</tspan><tspan x="87" dy="19">expiry == now</tspan><tspan x="87" dy="19">not just earlier</tspan></text></g><path d="M87 237L87 275" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M83.64402122977057,268.85692206676737L87,275L90.35597877022943,268.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="12" y="279" width="150" height="80" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="87" y="295.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="87" dy="0">Counterexample</tspan><tspan x="87" dy="19">fails</tspan><tspan x="87" dy="19">RETURN for</tspan><tspan x="87" dy="19">revision</tspan></text></g><path d="M162 319L198 319" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M191.8569220667674,322.3559787702294L198,319L191.8569220667674,315.6440212297706" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="198" y="279" width="150" height="80" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="273" y="305" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">Repair same</tspan><tspan x="273" dy="19">scope</tspan><tspan x="273" dy="19">rerun evidence</tspan></text></g><text x="180" y="407" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Acceptance stays with the engineer.</tspan></text><text x="180" y="437" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Fictional patch; passing tests alone is not</tspan><tspan x="180" dy="19">acceptance.</tspan></text></svg><div class="visual-derivation-caption">The independent equality case breaks the proposed patch. Return it for revision without widening scope. The agent supplies evidence; the engineer retains acceptance authority.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="agent-review-control-loop"><strong>Check:</strong> Can a passing agent-run test authorize acceptance? What independent evidence is still missing?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="agent-review-control-loop">Expiry equal to now must be rejected under this contract. A patch tested only before and after the boundary can miss it.</span></figcaption></figure>

For example:

> Inspect `metrics.py` and the merge test. Repair only `StreamingConfusion.merge` so merging two shards equals a single pass over their combined examples. Preserve the public API. Explain the defect before proposing a patch, then run the focused merge test.

The candidate should still inspect the patch. Models commonly make adjacent "cleanups," weaken tests, change edge semantics, or replace a bounded fix with a broad abstraction. The right response is not trust or distrust in the abstract. It is review against the contract.

## What an L4 answer sounds like

> "I would paste the repository into the agent, ask it to fix everything, run the tests, and then clean up whatever is left."

This can produce code, but it produces little hiring signal. There is no codebase model, no invariant, no bounded delegation, and no independent proof. A green suite may only mean the agent changed the suite.

## What an L5 answer adds

An L5 candidate maps the codebase, runs the failing test, and writes a bounded prompt. They inspect the diff, explain the merge bug, implement the report contract, and add a low-support slice test. They use the agent for local code generation or test enumeration but retain the architecture and metric semantics.

They also distinguish verification layers:

1. **Focused regression:** did the original bug disappear?
2. **Contract test:** does shard merge equal single-pass evaluation?
3. **Full suite:** did the change break an adjacent path?
4. **Behavioral check:** does the guardrail select the intended slice on a small hand-worked example?

This is reliable senior execution.

## What an L6 answer adds

An L6 candidate treats the interaction itself as part of the system. They choose delegation based on blast radius, make the agent expose assumptions before code, and preserve an evidence trail that another engineer can review.

They notice second-order risks:

- A mathematically mergeable metric may still be operationally non-mergeable if class vocabularies or thresholds differ across shards.
- A weakest-slice guardrail needs minimum support and a stable tie rule, or it will alert on noise.
- Tests can validate examples while missing data-contract drift at ingestion.
- An agent asked to optimize broadly may alter public semantics while preserving visible tests.
- Prompting consumes time. For a three-line fix, direct coding may be faster and easier to audit.

They finish by stating residual risk and the next check. They do not claim that passing public tests proves production safety.

## Tells that get you a strong-hire vote

- You map the codebase before asking the agent to act.
- You state a domain invariant, not merely a failing test.
- Prompts are bounded by file, behavior, and verification.
- You reject or narrow at least one generated change.
- You add an edge test the agent missed.
- You can explain every final line and every metric semantic.
- You end with residual risk and a next diagnostic.

## Tells that get you down-leveled

- The first action is "fix the repository."
- You accept a large diff because the suite is green.
- You let the agent change tests without proving the old expectation was wrong.
- You narrate prompts but cannot explain the generated algorithm.
- You optimize prompt style while ignoring the metric contract.
- You use AI for a task that would be faster and safer to write directly.

## Common follow-up

"The agent produced a correct patch faster than you could. Why does your review still matter?"

Because correctness is conditional on the contract the agent saw. The reviewer knows which requirements were omitted, which tests are incomplete, and which production invariants sit outside the repository. Speed to a plausible patch is useful. Acceptance authority still belongs to the engineer who can connect the patch to system behavior.

*Related: [debug a frontier LLM training run](/questions/debug-frontier-llm-training-run/), [mergeable streaming metrics](/questions/implement-streaming-classification-metrics/), and the [agentic codebase lab](/prep/labs/agentic-codebase/).*
