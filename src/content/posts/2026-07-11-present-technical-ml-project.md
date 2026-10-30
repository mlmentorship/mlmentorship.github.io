---
title: "Present a technical ML project and defend it under questions"
description: "Present owned decisions with credible alternatives, prior evidence, technical depth and honest outcome limits, even when questions change the order."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

Present a technical ML project through its problem, your decisions, supporting evidence, and outcomes while handling detailed follow-up questions.

## Problem

**Context.** An interviewer requests a project you personally worked on; the fictional critical-path trace below is a separate presentation rehearsal.

**Task.** Explain the goal, ownership, constraints, mechanism, alternatives, validation, and outcome of your project.
For the rehearsal, choose the first dependency to investigate and state what one trace cannot establish.

**Constraints.** Distinguish your contribution from the team's and observed results from estimates or counterfactuals.
Do not claim the fictional example as personal experience or infer a population p99 from one request trace.

In this fictional rehearsal, the complete request has a 100 ms p99 objective.
One critical-path trace starts at 0 ms:

| Parallel dependency | Completion |
| --- | ---: |
| Model inference | 45 ms |
| Feature join | 120 ms |

The response needs both results; other stages are omitted from this simplified trace.
Which component should you investigate first? What does this trace not establish?

## Picture: A Question Opens One Decision Record

<!-- visual:technical-project-decision-spine -->
<figure class="learning-figure" aria-labelledby="technical-project-decision-spine-title">
	<p class="visual-kicker">Decision records</p>
	<p class="visual-title" id="technical-project-decision-spine-title">A faster model cannot bypass the slower dependency that the response awaits</p><div class="visual-argument" data-visual-argument="technical-project-decision-spine" data-argument-kind="bars"><p class="visual-argument-premise">The awaited dependency can dominate a faster model</p><svg viewBox="0 0 360 277" role="img" aria-label="The awaited dependency can dominate a faster model. A response waits for parallel dependencies A=100 and model B=60. Halving model time leaves the awaited 100-unit dependency unchanged."><title>The awaited dependency can dominate a faster model</title><desc>A response waits for parallel dependencies A=100 and model B=60. Halving model time leaves the awaited 100-unit dependency unchanged.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">End-to-end before: max(100,60)</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="100" data-proof-min="0" data-proof-max="100" data-proof-zero="28" x="28" y="46" width="304" height="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><text x="28" y="106" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="28" dy="0">100 toy time units</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">End-to-end after: max(100,30)</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="100" data-proof-min="0" data-proof-max="100" data-proof-zero="28" x="28" y="163" width="304" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><text x="28" y="223" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="28" dy="0">100 toy time units</tspan></text><text x="28" y="257" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="257" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">100</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Chronological project tour compared with three self-contained decision records">
		<section class="visual-panel">
			<h4>Chronology alone</h4>
			<p><strong>Setup &rarr; data &rarr; model &rarr; launch</strong><br />This order can hide ownership and alternatives. A question may require reconstructing an earlier decision.</p>
		</section>
		<section class="visual-panel">
			<h4>Independent records</h4>
			<p><strong>Decision 1 &middot; Decision 2 &middot; Decision 3</strong><br />Each carries constraint, alternatives, prior evidence, choice/owner, result and update. Open the requested record directly.</p>
		</section>
	</div>
	<section class="visual-panel" role="group" aria-label="One simplified request trace, not a latency distribution">
		<h4>One request, two parallel dependencies</h4>
		<svg viewBox="0 0 330 180" role="img" aria-labelledby="project-trace-title project-trace-desc">
			<title id="project-trace-title">The join, not the model, blocks this request</title>
			<desc id="project-trace-desc">Both dependencies start at zero. The model completes at 45 milliseconds and the feature join at 120. The response waits for both, so its completion cannot precede 120 milliseconds. A line at 100 marks the objective value, not a measured percentile. Other work is omitted. Horizontal coordinates use x equals 70 plus two times milliseconds.</desc>
			<rect class="viz-plot-bg" x="4" y="4" width="322" height="172" rx="4"></rect>
			<path d="M270 24V124" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:5 4"></path>
			<text class="viz-label" x="70" y="35">model · 45 ms</text>
			<rect class="viz-node viz-node--input" x="70" y="44" width="90" height="20" rx="2"></rect>
			<text class="viz-label" x="70" y="85">feature join · 120 ms</text>
			<rect class="viz-node viz-node--warning" x="70" y="94" width="240" height="20" rx="2"></rect>
			<path class="viz-axis" d="M70 124H320"></path>
			<text class="viz-label" x="70" y="140" text-anchor="middle">0</text>
			<text class="viz-label" x="270" y="140" text-anchor="middle">100</text>
			<text class="viz-label" x="310" y="140" text-anchor="middle">120 ms</text>
			<text class="viz-label" x="270" y="161" text-anchor="middle">100 ms objective</text>
		</svg>
	</section>
	<div class="visual-derivation-caption">open the relevant decision, then its supporting trace. The join blocks this request; neither bar estimates the complete request's p99.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="technical-project-decision-spine"><strong>Check:</strong> Why can making the 45ms model faster not by itself bring this request below the 100ms objective?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="technical-project-decision-spine">A response waits for parallel dependencies A=100 and model B=60. Halving model time leaves the awaited 100-unit dependency unchanged.</span></figcaption>
</figure>

## Investigate the Blocker Without Overclaiming

Completion cannot precede $\max(45,120)=120$ ms in this trace.
Making inference instantaneous would still leave the join at 120 ms.

Investigate a versioned join change under matched load and inspect complete request latency and quality.
A single trace does not estimate p99; stage percentiles must not be added into an end-to-end claim.
State your actual ownership and other teams' changes.

## The Decision Spine

Use this editable record for each real consequential choice:

| Field | Record |
| --- | --- |
| Constraint | What blocked the obvious approach? |
| Alternatives | Which credible options existed? |
| Evidence then | What was known before the outcome? |
| Choice/owner | What did you decide, and with whose authority? |
| Result | What happened, including limits or actual surprises? |
| Update | What changed, or would change under new evidence? |

Connect one record to a sanitized calculation, trace, ablation, data contract or invariant.
Do not invent failure, metrics, ownership or a causal interpretation to fill fields.

## Change 1: Ten Minutes Remain

**New task.** Time shrinks or questions interrupt early.

**Picture and change.** Keep the opening claim and one complete record; place other evidence in appendices.

**Check.** Remove repetition, not the reason for the choice. Answer directly before returning to the deck.

**Use when.** Delivery time or question order changes.

## Change 2: Resources or Criticism Change

**New task.** Compute is halved or a credible critic disputes the reported effect.

**Picture and change.** Re-evaluate constraints and evidence; identify which intervention to retain and which claim changes.

**Check.** Do not defend historical architecture after its assumptions fail.

**Use when.** A changed condition tests your model of the system.

## Failure: Impressive Metrics Without an Owned Choice

An architecture tour cannot substitute for a credible alternative, owner, supporting mechanism and evidence limit.

## Transfer Check

Why does a holdback alone not prove your impact claim?

### Reasoned Answer

Causal interpretation depends on assignment, exposure, measurement, interference, attrition and analysis assumptions.
Name the actual design and limits.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Choose the project with the best evidence

Check five properties: owned consequential judgment, credible alternatives, actual uncertainty/surprises, outcome evidence or explicit limits, and implementation depth.
Prestige, duration and a compulsory failure story are not substitutes. Respect confidentiality and approved disclosure boundaries.

## Open with the answer

State problem, importance, personal responsibility, observed result/limits and the decision roadmap.
A two-minute opening, three records and 30-minute talk plus 15-minute questions are configurable rehearsal settings, not universal interview facts.

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-adds"></span>
<span id="what-an-l6-answer-adds"></span>
## Evidence Ledger

| Claim | Source/version | Personal action | Collaborator | Observed outcome | Causal design/limit | Changed condition |
| --- | --- | --- | --- | --- | --- | --- |
| Fill with approved real evidence | Exact artifact/trace | Your contribution | Other owners | Measured or unavailable | Identification assumptions | Reopening evidence |

Use the same names at overview, architecture and implementation depth.
Explain adoption, incentives and second-order effects only when supported; distinguish your contribution from the project's total result.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
## Check the Account

Can a question enter through evidence, ownership, mechanism or changed constraints without losing the claim?
If evidence is unavailable or confidential, state the limit rather than inventing a number.

<span id="tells-that-get-you-down-leveled"></span>
## Missing Evidence

Replace vague team attribution, offline-as-product claims and unsupported strategic language with the actual decision record.

## Common follow-up

What would your strongest critic say was wrong? State the actual criticism, supporting evidence, agreement/disagreement and decision to revisit.
Do not manufacture a critic or substitute a disguised compliment.

Use the [presentation protocol](/prep/presentation/) and attempt it before reading this page again.

## Sources

- [Garner and Alley, assertion-evidence presentation study](https://pure.psu.edu/en/publications/how-the-design-of-presentation-slides-affects-audience-comprehens/).
- [OPM structured-interview guide](https://www.opm.gov/policy-data-oversight/assessment-and-selection/structured-interviews/guide.pdf).

*Related: [your most ambitious project](/questions/most-ambitious-project/), [scope an ambiguous problem](/questions/scope-ambiguous-problem/), and [technical presentation practice](/prep/presentation/).*

</div>
