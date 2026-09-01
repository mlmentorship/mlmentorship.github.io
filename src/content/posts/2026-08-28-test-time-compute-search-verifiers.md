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

**Learning objective:** separate proposal coverage from selection quality by explaining how extra compute can generate a correct candidate yet still return a wrong answer when the verifier ranks a proxy instead of correctness.

<!-- visual:test-time-compute-verifier-bottleneck -->
<figure class="learning-figure" aria-labelledby="test-time-verifier-heading">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="test-time-verifier-heading">More candidates help only if the verifier selects the right one.</p>
	<svg viewBox="0 0 360 410" role="img" aria-labelledby="test-time-verifier-title test-time-verifier-desc">
		<title id="test-time-verifier-title">The same candidate set succeeds with a direct checker and fails with a weak proxy verifier</title>
		<desc id="test-time-verifier-desc">Two panels contain the same four sampled candidates: A has wrong arithmetic, B is correct, C is incomplete, and D is a polished shortcut with a wrong answer. In the upper panel, protected tests directly check the task and select candidate B, so the returned answer is correct. In the lower panel, a weak model judge rewards plausibility and selects candidate D, so the returned answer is wrong even though candidate B was available. Candidate letters, truth labels, verifier labels, selection text, and solid versus dashed arrows communicate the result without color.</desc>
		<defs>
			<marker id="test-time-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="viz-arrow-forward"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="27" width="344" height="174" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="18">1 · DIRECT CHECK: COVERAGE + SELECTION BOTH SUCCEED</text>
		<text class="viz-label" x="18" y="48">SAME FOUR SAMPLES</text>
		<rect class="viz-node" x="18" y="59" width="82" height="43" rx="4"></rect>
		<rect class="viz-node viz-node--output" x="108" y="59" width="82" height="43" rx="4"></rect>
		<rect class="viz-node" x="18" y="110" width="82" height="43" rx="4"></rect>
		<rect class="viz-node" x="108" y="110" width="82" height="43" rx="4"></rect>
		<text class="viz-label" x="59" y="77" text-anchor="middle">A · wrong</text><text class="viz-label" x="59" y="92" text-anchor="middle">arithmetic</text>
		<text class="viz-callout" x="149" y="77" text-anchor="middle">B · CORRECT</text><text class="viz-label" x="149" y="92" text-anchor="middle">solution</text>
		<text class="viz-label" x="59" y="132" text-anchor="middle">C · incomplete</text>
		<text class="viz-label" x="149" y="128" text-anchor="middle">D · polished</text><text class="viz-label" x="149" y="143" text-anchor="middle">wrong shortcut</text>
		<path d="M198 105H216" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#test-time-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="222" y="59" width="118" height="43" rx="4"></rect>
		<text class="viz-axis-label" x="281" y="76" text-anchor="middle">PROTECTED TESTS</text>
		<text class="viz-label" x="281" y="92" text-anchor="middle">check task outcome</text>
		<path d="M281 103V119" fill="none" stroke="var(--viz-output-stroke)" stroke-width="3" marker-end="url(#test-time-arrow)"></path>
		<rect class="viz-node viz-node--output" x="222" y="127" width="118" height="47" rx="4"></rect>
		<text class="viz-callout" x="281" y="146" text-anchor="middle">SELECT B · PASS</text>
		<text class="viz-label" x="281" y="163" text-anchor="middle">return correct answer</text>
		<text class="viz-label" x="18" y="187">Sampling found B; direct evidence preserved it.</text>
		<rect class="viz-plot-bg" x="8" y="229" width="344" height="174" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="220">2 · PROXY CHECK: COVERAGE SUCCEEDS, SELECTION FAILS</text>
		<text class="viz-label" x="18" y="250">SAME FOUR SAMPLES</text>
		<rect class="viz-node" x="18" y="261" width="82" height="43" rx="4"></rect>
		<rect class="viz-node viz-node--output" x="108" y="261" width="82" height="43" rx="4"></rect>
		<rect class="viz-node" x="18" y="312" width="82" height="43" rx="4"></rect>
		<rect class="viz-node viz-node--warning" x="108" y="312" width="82" height="43" rx="4"></rect>
		<text class="viz-label" x="59" y="279" text-anchor="middle">A · wrong</text><text class="viz-label" x="59" y="294" text-anchor="middle">arithmetic</text>
		<text class="viz-callout" x="149" y="279" text-anchor="middle">B · CORRECT</text><text class="viz-label" x="149" y="294" text-anchor="middle">solution</text>
		<text class="viz-label" x="59" y="334" text-anchor="middle">C · incomplete</text>
		<text class="viz-label" x="149" y="330" text-anchor="middle">D · polished</text><text class="viz-label" x="149" y="345" text-anchor="middle">wrong shortcut</text>
		<path d="M198 307H216" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#test-time-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="222" y="261" width="118" height="43" rx="4"></rect>
		<text class="viz-axis-label" x="281" y="278" text-anchor="middle">WEAK MODEL JUDGE</text>
		<text class="viz-label" x="281" y="294" text-anchor="middle">rewards plausibility</text>
		<path d="M281 305V321" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="3" stroke-dasharray="6 4" marker-end="url(#test-time-arrow)"></path>
		<rect class="viz-node viz-node--warning" x="222" y="329" width="118" height="47" rx="4"></rect>
		<text class="viz-callout" x="281" y="348" text-anchor="middle">SELECT D · FAIL</text>
		<text class="viz-label" x="281" y="365" text-anchor="middle">return wrong answer</text>
		<text class="viz-label" x="18" y="389">Sampling still found B; the proxy discarded it.</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> candidate B exists in both panels, so generation coverage is identical. Protected tests select B by checking the outcome; a plausibility judge selects polished-but-wrong D. More samples cannot rescue a selection rule that rewards the wrong thing.</figcaption>
</figure>

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