---
title: "Extend an unfamiliar ML codebase with an AI coding agent"
description: "The new coding signal is control: map the system, delegate bounded work, reject bad changes, and prove the result."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> You inherit a small ML evaluation service with failing tests. Fix a distributed metric bug, add a slice guardrail, and use the provided coding agent where it helps.

The strongest candidate does not start by prompting. They first map the execution path, state the invariant, and decide which work is safe to delegate. The agent accelerates implementation; the candidate owns the plan, review, and proof.

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
<figure class="learning-figure" aria-labelledby="agent-control-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="agent-control-title">Separate delegated execution from acceptance authority.</p>
	<div class="visual-grid--two" role="group" aria-label="Two-lane workflow comparing work delegated to a coding agent with decisions retained by the engineer">
		<section class="visual-panel">
			<h4>Agent lane · bounded execution</h4>
			<p>The agent proposes evidence inside the stated boundary.</p>
			<ol>
				<li><strong>Inspect:</strong> trace the named files and failing path.</li>
				<li><strong>Patch:</strong> change only the allowed surface.</li>
				<li><strong>Exercise:</strong> run the requested focused test.</li>
				<li><strong>Report:</strong> expose the diff, output, and assumptions.</li>
			</ol>
		</section>
		<section class="visual-panel">
			<h4>Reviewer lane · acceptance authority</h4>
			<p>The engineer owns the contract before and after delegation.</p>
			<ol>
				<li><strong>Frame:</strong> map the path and state the invariant.</li>
				<li><strong>Inspect:</strong> reject scope drift and weakened tests.</li>
				<li><strong>Prove:</strong> add an independent contract or edge test.</li>
				<li><strong>Decide:</strong> accept only with residual risk stated.</li>
			</ol>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> the agent can inspect, patch, and run a focused test, but those outputs are evidence rather than acceptance. The engineer sets the invariant, reviews the actual diff, supplies independent proof, and retains the final merge decision.</figcaption>
</figure>

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
