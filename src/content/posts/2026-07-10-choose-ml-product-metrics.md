---
title: "Choose metrics for an ML product"
description: "Define metric populations, horizons and decision roles, then apply a joint release rule without overriding its guardrails."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

Choose metrics and a release decision that account for user outcomes, cost, uncertainty, and mandatory operating limits.

## Problem

**Context.** An invented randomized experiment compares two product versions on independent eligible users, each with one binary task outcome.
The results and predeclared quality, cost, and latency rules appear below.

**Task.** Compute the improvement and cost per success, assess the practical-effect rule, and apply every release gate.
Explain which measurements are primary outcomes, guardrails, or diagnostics.

**Constraints.** Preserve the stated unit of analysis and predeclared thresholds.
A positive headline result cannot override a failed mandatory guardrail; distinguish observed estimates from confidence guarantees.

An invented randomized, independent-user experiment has 10,000 eligible users per arm, each contributing one binary task outcome.
Arm A completes 6,000 tasks for $10,000; B completes 6,600 for $13,000.

Predeclared rules require improvement's lower confidence bound above two percentage points and cost/success below $2.
The p95 latency limit is two seconds.
B's observed p95 is 2.1 seconds.
Should the full release proceed?

## Picture

The release rule is an AND gate, not a vote among favorable measurements.

<!-- visual:product-metric-decision-authority -->
<figure class="learning-figure editorial-illustration" aria-labelledby="product-metric-decision-authority-edition-title"><span class="visual-fragment-anchor" id="metric-authority-title" aria-hidden="true"></span><p class="visual-title" id="product-metric-decision-authority-edition-title">Two passing metrics cannot overrule the failed latency limit</p><div class="visual-argument" data-visual-argument="product-metric-decision-authority" data-argument-kind="matrix"><p class="visual-argument-premise">Two passing gates cannot outvote the failed third gate</p><svg viewBox="0 0 360 189" role="img" aria-label="Two passing gates cannot outvote the failed third gate. Completion and cost pass, but latency exceeds its declared limit. The release condition is an AND, not a majority vote."><title>Two passing gates cannot outvote the failed third gate</title><desc>Completion and cost pass, but latency exceeds its declared limit. The release condition is an AND, not a majority vote.</desc><text x="126.48333333333335" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="126.48333333333335" dy="0">observed</tspan></text><text x="211.89000000000001" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="211.89000000000001" dy="0">required</tspan></text><text x="297.29666666666674" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="297.29666666666674" dy="0">decision</tspan></text><text x="71.78" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">gain</tspan></text><rect data-proof-cell="0,0" x="83.78" y="36" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="124.98333333333335" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="124.98333333333335" dy="0">4.66</tspan></text><rect data-proof-cell="0,1" x="169.18666666666667" y="36" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="210.39000000000001" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="210.39000000000001" dy="0">&gt;=2</tspan></text><rect data-proof-cell="0,2" x="254.59333333333336" y="36" width="82.40666666666668" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="295.7966666666667" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="295.7966666666667" dy="0">pass</tspan></text><text x="71.78" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">cost</tspan></text><rect data-proof-cell="1,0" x="83.78" y="83" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="124.98333333333335" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="124.98333333333335" dy="0">1.97</tspan></text><rect data-proof-cell="1,1" x="169.18666666666667" y="83" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="210.39000000000001" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="210.39000000000001" dy="0">&lt;=2</tspan></text><rect data-proof-cell="1,2" x="254.59333333333336" y="83" width="82.40666666666668" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="295.7966666666667" y="110.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="295.7966666666667" dy="0">pass</tspan></text><text x="71.78" y="148" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">latency</tspan></text><rect data-proof-cell="2,0" x="83.78" y="130" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="124.98333333333335" y="157.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="124.98333333333335" dy="0">2.1</tspan></text><rect data-proof-cell="2,1" x="169.18666666666667" y="130" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="210.39000000000001" y="157.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="210.39000000000001" dy="0">&lt;=2</tspan></text><rect data-proof-cell="2,2" x="254.59333333333336" y="130" width="82.40666666666668" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="295.7966666666667" y="157.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="295.7966666666667" dy="0">FAIL</tspan></text><path d="M258.59333333333336 170L333.00000000000006 134" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 535" role="img" aria-labelledby="product-metric-decision-authority-edition-svg-title product-metric-decision-authority-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="product-metric-decision-authority-edition-svg-title">Two passing metrics cannot cancel one latency veto</title><desc id="product-metric-decision-authority-edition-svg-desc">The completion gain lower bound 4.66 points exceeds its two-point floor, cost 1.97 dollars is below two dollars, but latency 2.1 seconds exceeds two seconds. Each metric has its own labelled scale and all three feed an AND gate. Diagnostics cannot authorize release.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Declared gates; each row has its own scale</tspan></text><text x="18" y="60" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Completion gain: lower bound</tspan></text><path d="M30 98L330 98" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M130 83L130 115" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="263" cy="98" r="6" style="fill:var(--viz-output-stroke)"></circle><text x="30" y="142" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="30" dy="0">observed 4.66 points</tspan></text><text x="330" y="142" style="font-size:12px;font-weight:400;fill:var(--viz-output-stroke);text-anchor:end"><tspan x="330" dy="0">PASS; limit 2</tspan></text><text x="18" y="172" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Cost per completed task</tspan></text><path d="M30 210L330 210" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M270 195L270 227" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="266.4" cy="210" r="6" style="fill:var(--viz-output-stroke)"></circle><text x="30" y="254" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="30" dy="0">observed 1.97 dollars</tspan></text><text x="330" y="254" style="font-size:12px;font-weight:400;fill:var(--viz-output-stroke);text-anchor:end"><tspan x="330" dy="0">PASS; limit 2</tspan></text><text x="18" y="284" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Latency</tspan></text><path d="M30 322L330 322" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M270 307L270 339" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="282" cy="322" r="6" style="fill:var(--viz-warning-stroke)"></circle><text x="30" y="366" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="30" dy="0">observed 2.1 seconds</tspan></text><text x="330" y="366" style="font-size:12px;font-weight:400;fill:var(--viz-warning-stroke);text-anchor:end"><tspan x="330" dy="0">FAIL; limit 2</tspan></text><g data-editorial-node><rect x="30" y="407" width="300" height="58" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="180" y="431.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">AND gate: 2 passes + 1 failure</tspan><tspan x="180" dy="19">HOLD full release</tspan></text></g><text x="180" y="499" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Retries and edits can diagnose; they cannot</tspan><tspan x="180" dy="19">override.</tspan></text></svg><div class="visual-derivation-caption">Use each row's units and declared inequality. Completion and cost pass; latency fails. Their conjunction holds full release. Diagnostics explain the failure but cannot override it.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="product-metric-decision-authority"><strong>Check:</strong> Which failed comparison blocks release despite favorable completion and cost?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="product-metric-decision-authority">Completion and cost pass, but latency exceeds its declared limit. The release condition is an AND, not a majority vote.</span></figcaption></figure>

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
