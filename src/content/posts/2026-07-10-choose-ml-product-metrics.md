---
title: "Choose metrics for an ML product"
description: "Define metric populations, horizons and decision roles, then apply a joint release rule without overriding its guardrails."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

Give each metric a decision role: primary outcomes support release, guardrails can veto, and diagnostics explain behavior.

## Problem

An invented randomized, independent-user experiment has 10,000 eligible users per arm, each contributing one binary task outcome.
Arm A completes 6,000 tasks for $10,000; B completes 6,600 for $13,000.

Predeclared rules require improvement's lower confidence bound above two percentage points and cost/success below $2.
The p95 latency limit is two seconds.
B's observed p95 is 2.1 seconds.
Should the full release proceed?

## Picture

The release rule is an AND gate, not a vote among favorable measurements.

<!-- visual:product-metric-decision-authority -->
<figure class="learning-figure" aria-labelledby="metric-authority-title">
	<p class="visual-kicker">Quality passes; latency vetoes</p>
	<p class="visual-title" id="metric-authority-title">A favorable completion interval cannot override a failed latency gate.</p>
	<div class="visual-grid--two" role="group" aria-label="Product metrics grouped by their authority in a launch decision">
		<section class="visual-panel">
			<h4>DECISION GATE · CAN AUTHORIZE OR VETO</h4>
			<p><strong>Completion: pass</strong><br />Improvement 6 points; approximate lower bound 4.66 points exceeds the two-point floor.</p>
			<p><strong>Cost/success: pass</strong><br />B costs about $1.97 per completed task, below $2.</p>
			<p><strong>Latency: fail</strong><br />B's 2.1 seconds exceeds two seconds. Do not authorize full release.</p>
		</section>
		<section class="visual-panel">
			<h4>SUPPORTING EVIDENCE · CANNOT DECLARE VICTORY</h4>
			<p><strong>Diagnostics: explain</strong><br />Retries, edits, abandonment and escalation help locate the change.</p>
			<p><strong>Offline metrics: qualify</strong><br />Model and slice checks support iteration, not proof of user value.</p>
			<p><strong>No override</strong><br />Supporting signals cannot cancel a failed release gate.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> completion and cost pass their declared rules, but latency fails. Diagnose latency or justify a different eligible treatment before testing again.</figcaption>
</figure>

## Calculate the decision

Completion changes from $0.60$ to $0.66$, a six-percentage-point increase.
At a fixed readout, an approximate independent-binomial 95% interval is:

$$
SE=\sqrt{\frac{0.60(0.40)+0.66(0.34)}{10000}}\approx0.006815,
\qquad 0.06\pm1.96SE\approx[0.04664,0.07336].
$$

Costs per success are $10000/6000\approx1.67$ and $13000/6600\approx1.97$ dollars.
These hypothetical policy margins are not statistical estimates or universal product rules.

## Change the horizon or harm distribution

### 1. Confirm resolution after seven days

**Picture and change:** exclude immature cohorts from final resolution rates, or model censoring explicitly.
Keep immediate acceptance diagnostic.

**Check:** yesterday's acceptance cannot establish seven-day resolution.

**Use when:** value becomes observable after the interaction.

### 2. A critical slice degrades

**Picture and change:** predeclare a consequential-slice safety gate and qualified review process.

**Check:** an overall gain cannot override that failed gate.

**Use when:** rare severe harms differ from ordinary dissatisfaction, including medical-use evaluation.

## Failure and transfer

Longer sessions caused by repeated failures are not evidence of greater user value.

Completion rises but seven-day retention falls. Which wins?

<details>
<summary>Check your answer</summary>

Apply predeclared roles, horizon, uncertainty and veto margins; do not choose a favorable hierarchy afterward.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Start with the product decision

Identify the user, job, controllable behavior, unacceptable failures and time to value.
Define numerator, denominator, eligibility, unit, window, direction, uncertainty, role, threshold, owner and review date.
Repeated tasks per user require a compatible uncertainty method, not the independent-user calculation above.

## A metric hierarchy

Choose a primary outcome close to measurable value: completion, qualified resolution, retention or incremental conversion.
Protect safety, latency, availability, cost/success and relevant ecosystem effects.
Keep acceptance, edits, retries, tool calls, fallbacks, retrieval and calibration diagnostic.

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-adds"></span>
<span id="what-an-l6-answer-adds"></span>

## Explain scope, not a level label

A metric list is incomplete without populations, decisions and tradeoffs.
Explain delayed effects and user adaptation; faster completion can undermine learning or trust.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Review gaming and measurement bias

Acceptance can rise when users stop checking.
Thumbs-up data can have response and selection bias.
No metric list guarantees a hiring outcome.

## Common follow-ups

State the planned retention rule when completion rises.
Use mature cohorts for delayed labels.
Test whether engagement rewards retries.
Choose mandatory slices from plausible harms.
Healthcare applications need qualified review and distinct severity gates, not an averaged satisfaction score.

*Related: [expected calibration error](/concepts/expected-calibration-error/), [A/B testing for ML](/concepts/ab-testing-for-ml/), and [evaluate an LLM application](/questions/how-would-you-evaluate-an-llm-application/).*

</div>
