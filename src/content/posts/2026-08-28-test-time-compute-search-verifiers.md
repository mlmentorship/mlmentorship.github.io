---
title: "Test-time compute, search, and verifiers"
description: "Spend extra inference compute on several candidate solutions, search, tools, or revision, then select results with evidence."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Test-time compute adds sampling, search, tools or revision while answering a request; selection must retain any correct candidate found.

For nonnegative integer $n$, a task asks for $1+\cdots+n$.
Four hypothetical programs produce these outputs on three checks:

| Candidate | Rule | $n=0$ | $n=3$ | $n=10$ |
| --- | --- | --- | --- | --- |
| A | $n(n-1)/2$ | $0$ | $3$ | $45$ |
| B | $n(n+1)/2$ | $0$ | $6$ | $55$ |
| C | Only handles $3$ | Missing | $6$ | Missing |
| D | Always returns $6$ | $6$ | $6$ | $6$ |

Expected outputs are `[0, 6, 55]`.
A plausibility scorer assigns A/B/C/D scores `[0.60, 0.80, 0.40, 0.95]`.
Which program does each selector return?

Hold proposal coverage fixed: B exists in both panels.

<!-- visual:test-time-compute-verifier-bottleneck -->
<figure class="learning-figure" aria-labelledby="test-time-verifier-heading">
	<p class="visual-kicker">Same candidates; different selection evidence</p>
	<p class="visual-title" id="test-time-verifier-heading">Generating a correct candidate is insufficient when the verifier discards it</p><div class="visual-argument" data-visual-argument="test-time-compute-verifier-bottleneck" data-argument-kind="matrix"><p class="visual-argument-premise">Generating a correct answer is insufficient if the verifier rejects it</p><svg viewBox="0 0 360 142" role="img" aria-label="Generating a correct answer is insufficient if the verifier rejects it. Candidate A is correct but discarded; B is wrong but selected. More generation alone does not repair this selection bottleneck."><title>Generating a correct answer is insufficient if the verifier rejects it</title><desc>Candidate A is correct but discarded; B is wrong but selected. More generation alone does not repair this selection bottleneck.</desc><text x="128.5" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="128.5" dy="0">correct</tspan></text><text x="269.5" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="269.5" dy="0">selected</tspan></text><text x="46" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">A</tspan></text><rect data-proof-cell="0,0" x="58" y="36" width="138" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="127" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="127" dy="0">yes</tspan></text><rect data-proof-cell="0,1" x="199" y="36" width="138" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="268" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="268" dy="0">NO</tspan></text><path d="M203 76L333 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="46" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">B</tspan></text><rect data-proof-cell="1,0" x="58" y="83" width="138" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="127" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="127" dy="0">no</tspan></text><rect data-proof-cell="1,1" x="199" y="83" width="138" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="268" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="268" dy="0">yes</tspan></text><path d="M203 123L333 87" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 410" role="img" aria-labelledby="test-time-verifier-title test-time-verifier-desc">
		<title id="test-time-verifier-title">The same candidate set succeeds with a direct checker and fails with a weak proxy verifier</title>
		<desc id="test-time-verifier-desc">Two panels contain the same four sampled candidates: A has wrong arithmetic, B is correct, C is incomplete, and D is a polished shortcut with a wrong answer. In the upper panel, protected tests directly check the task and select candidate B, so the returned answer is correct. In the lower panel, a weak model judge rewards plausibility and selects candidate D, so the returned answer is wrong even though candidate B was available. Candidate letters, truth labels, verifier labels, selection text, and solid versus dashed arrows communicate the result without color.</desc>
		<defs>
			<marker id="test-time-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="viz-arrow-forward"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="27" width="344" height="174" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="18">1 · DIRECT CHECK: FINDS AND SELECTS B</text>
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
		<text class="viz-callout" x="281" y="146" text-anchor="middle">SELECT B · 3/3</text>
		<text class="viz-label" x="281" y="163" text-anchor="middle">return correct answer</text>
		<text class="viz-label" x="18" y="187">Sampling found B; direct evidence preserved it.</text>
		<rect class="viz-plot-bg" x="8" y="229" width="344" height="174" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="220">2 · PROXY CHECK: FINDS B, SELECTS D</text>
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
	<div class="visual-derivation-caption">sampling already found B. The three checks retain B; the proxy discards it for D, which passes only one check.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="test-time-compute-verifier-bottleneck"><strong>Check:</strong> Was candidate generation the failure in the lower panel? What evidence caused the selector to discard the correct candidate?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="test-time-compute-verifier-bottleneck">Candidate A is correct but discarded; B is wrong but selected. More generation alone does not repair this selection bottleneck.</span></figcaption>
</figure>

## The verifier is the bottleneck

The following executable fixture checks the actual functions, not their written explanations.
These illustrative tests are finite evidence, not a proof for every input.

```python
proposals = {
	"A": lambda count: count * (count - 1) // 2,
	"B": lambda count: count * (count + 1) // 2,
	"C": lambda count: 6 if count == 3 else None,
	"D": lambda count: 6,
}
checks = [(0, 0), (3, 6), (10, 55)]
proxy_scores = {"A": 0.60, "B": 0.80, "C": 0.40, "D": 0.95}

def choose_checked(candidates, cases):
	passing = [
		name for name, function in candidates.items()
		if all(function(count) == expected for count, expected in cases)
	]
	return passing[0] if passing else None

assert choose_checked(proposals, checks) == "B"
assert max(proxy_scores, key=proxy_scores.get) == "D"
without_correct = {name: function for name, function in proposals.items() if name != "B"}
assert choose_checked(without_correct, checks) is None
```

For a selector returning one candidate:

$$
P(\text{return correct})=
P(\text{any correct})\,
P(\text{select correct}\mid\text{any correct}).
$$

The proxy fails selection, so more proposals alone do not repair this demonstrated fault.
Use reliable direct checks where possible; validate model judges and test false accepts, false rejects and unusual valid solutions.

## Change the sample budget or verification result

### 1. Increase from four candidates to eight

**Picture and change:** under hypothetical independent success probability $p=0.25$, coverage is $1-(1-p)^m$ for $m$ samples.
Retain a reliable selector and account for its cost too.

**Check:** coverage rises from $0.68359375$ to $0.899887085$.
Perfectly correlated samples remain at $0.25$ coverage.

A separate model-verifier accounting fixture charges 100 generation tokens and 50 verification tokens per candidate.
Four candidates cost 600 tokens; eight cost 1,200, versus 150 for one.
These are invented charges, not measured usage; Python execution itself does not imply model tokens.

**Use when:** coverage, not selection, limits quality; compare equal-cost alternatives, including a stronger base model.

### 2. B is absent and the budget is exhausted

**Picture and change:** return an explicit unverified/escalation result instead of silently selecting the highest proxy score.

**Check:** `choose_checked(without_correct, checks)` returns `None`; A/C/D each pass only one check.

**Use when:** unsupported answers cost more than abstention or qualified escalation.

## Failure and transfer

Protect evaluation from candidate modification; repeated feedback can overfit even hidden selection tests.
Keep separate confirmation data.

Does eight-way majority agreement establish correctness when every candidate repeats the same wrong formula?

<details>
<summary>Check your answer</summary>

No. Test a discriminating boundary or obtain independent evidence; repeated agreement is not independent correctness evidence.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Main methods

### Best-of-n sampling

Generate candidates and apply a selector; independence is an assumption for the coverage formula, not a sampling guarantee.

### Self-consistency

Aggregate answers; agreement can preserve correlated errors.

### Iterative revision

Revise against available evidence; unsupported critique can introduce errors.

### Search

Define states, expansion, score, pruning and stopping.
An early weak score can prune the only correct path.

### Tool use

Programs, symbolic solvers, environment checks and human review provide evidence only within their own assumptions.
Sandbox untrusted code and bound tool resources.

## Small example: coding assistant

Run focused tests, revise from failures, then run regressions and static checks.
Consider another plan for high risk; stop within scope or escalate missing requirements/checks.

## Allocate compute by request

Use task risk, difficulty, expected gain and remaining budget.
Disagreement or failed checks can trigger bounded escalation, not endless revision.

## Measure the compute-quality curve

Report success, verifier errors, tokens, tool calls, latency tails, cost and difficulty slices at completed budgets.
Compare equal total cost as well as sample count.

## Main failure modes

<span id="correlated-candidates"></span>
<span id="verifier-exploitation"></span>
<span id="search-error"></span>
<span id="unbounded-loops"></span>
<span id="latency-tails"></span>
<span id="hidden-quality-loss"></span>

Bound loops, inspect latency tails and overload behavior, and retain consequential held-out slices.

<span id="in-an-interview"></span>
<span id="common-mistakes"></span>
<span id="why-ai-labs-care"></span>

## Scope the answer

Define task, risk, method, verifier, allocation, stopping and measurement; do not infer correctness from additional compute.

References: [verifier training](https://arxiv.org/abs/2110.14168), [self-consistency](https://arxiv.org/abs/2203.11171), [process supervision](https://arxiv.org/abs/2305.20050), [allocation](https://arxiv.org/abs/2408.03314).

*Related: [train and serve a reasoning model under a fixed budget](/questions/design-reasoning-model-fixed-budget/), [annotated reasoning-strategy mock](/guides/annotated-reasoning-strategy-mock/), [evaluate an agent](/questions/evaluate-an-agent/), and [RL with verifiable rewards and GRPO](/concepts/rl-verifiable-rewards-grpo/).*

</div>