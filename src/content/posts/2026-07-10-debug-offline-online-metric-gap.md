---
title: "Your offline metric improved but the online metric got worse. Debug it."
description: "Separate experiment validity, delivered treatment and product outcomes before attributing an online regression to the model."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

Investigate why an offline model improvement accompanies an online product regression without assuming the model itself caused the gap.

## Problem

**Context.** A report compares offline and online changes, while the separate delivery fixture below includes users receiving fallback instead of treatment.

**Task.** Define the compared metrics, assess delivered treatment, and choose the next test that distinguishes measurement, serving, and modeling explanations.
State any immediate containment needed before investigating further.

**Constraints.** Preserve assignment denominators and separate assigned users from successfully served users.
Check relative versus absolute changes, label maturity, and metric meaning; the supplied aggregates alone do not identify a causal defect.

A report says offline quality improved 8% while the online primary metric declined 2%.
Check each metric's denominator, baseline and relative-change definition before comparing them.

In a separate hypothetical delivery fixture, A serves 1,000 assigned users and records 800 successes.
B delivers the new model to 900 users, recording 738 successes.
Its other 100 users receive fallback, with 40 successes.
Where should the next discriminating check focus?

## Picture

Keep assigned users in the product denominator while tracing delivery branches separately.

<!-- visual:offline-online-gap-diagnostic-order -->
<figure class="learning-figure editorial-illustration" aria-labelledby="offline-online-gap-diagnostic-order-edition-title"><p class="visual-title" id="offline-online-gap-diagnostic-order-edition-title">The missing 100 users change which arm wins</p><svg viewBox="0 0 360 430" role="img" aria-labelledby="offline-online-gap-diagnostic-order-edition-svg-title offline-online-gap-diagnostic-order-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="offline-online-gap-diagnostic-order-edition-svg-title">The missing 100 users change which arm wins</title><desc id="offline-online-gap-diagnostic-order-edition-svg-desc">Both illustrative arms contain 1,000 assigned users. A has 800 successes. B routes 900 users to the model, with 738 successes, and 100 to fallback, with 40 successes. The 900-to-100 strip is proportional. B therefore has 778 successes, 77.8 percent overall, despite 82 percent among model-delivered users.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative: 1,000 assigned users in each</tspan><tspan x="180" dy="19">arm</tspan></text><text x="20" y="61" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">A: 800 successes / 1,000 = 80%</tspan></text><rect x="25" y="82" width="310" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="102" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">all 1,000 assigned</tspan></text><text x="20" y="155" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">B: delivery is part of the treatment</tspan></text><rect x="25" y="177" width="279" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><rect x="304" y="177" width="31" height="29" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="154" y="197" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="154" dy="0">900 model</tspan></text><path d="M319 206L319 229" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="319" y="249" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="319" dy="0">100</tspan></text><text x="117" y="244" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="117" dy="0">738 successes</tspan></text><text x="275" y="278" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="275" dy="0">40 fallback successes</tspan></text><g data-editorial-node><rect x="30" y="305" width="300" height="58" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="339" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">(738 + 40) / 1,000 = 77.8%</tspan></text></g><text x="180" y="385" style="font-size:12px;font-weight:400;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">82% model-only != 77.8% assigned-user</tspan><tspan x="180" dy="19">result</tspan></text></svg><figcaption><strong>Read it this way:</strong> The denominator is everyone assigned, including fallback users. B has 738 + 40 successes, so 77.8% loses to 80%. These counts are illustrative. <span class="visual-reader-check" data-reader-check="offline-online-gap-diagnostic-order"><strong>Check:</strong> How can B's model succeed at 82% while B's assigned-user outcome is worse than A's 80%?</span></figcaption></figure>

## Calculate, then test a cause

$$
	ext{B overall}=\frac{738+40}{1000}=0.9(0.82)+0.1(0.40)=0.778.
$$

A's rate is $0.80$; the difference is $-0.022$, or $-2.2$ percentage points.
This is not the prompt's separate reported relative change.
The $0.82$ delivered-model rate conditions on routing and does not identify a causal model improvement.

## First: verify the observation

1. Contain harm under declared rollback conditions; preserve logs and versions.
2. Check assignment, exposure, sample ratios, instrumentation, uncertainty, delayed outcomes and novelty or seasonality.
3. Replay production inputs through the actual model, feature, candidate and preprocessing versions.
4. Inspect latency, errors, fallback and differences by user, item, time and surface.
5. After validity and delivery pass, test a specific model-quality or feedback hypothesis.

| Hypothesis | Predicted observation | Cheap discriminating check | Next action |
| --- | --- | --- | --- |
| Latency triggers fallback | Fallback rises on slow requests | Join routing reasons with latency/version records | Contain; repair and measure again |
| Preprocessor is stale | Same raw input yields different features | Compare versioned feature snapshots | Fix parity before retraining |

Record actual results and rollback conditions alongside the hypothesis.
These observations can localize delivery faults; they do not prove fallback is the only cause.
Restoring delivery needs a new controlled measurement of product effects.

## Change the population or label source

### 1. Only new users regress

**Picture and change:** compare feature defaults, candidate coverage and exposure using new-user traces.

**Check:** a replay matching returning users but failing new users localizes the next parity test.

**Use when:** a specific slice explains the aggregate change; do not jump directly to general drift.

### 2. Labels came from the old ranker

**Picture and change:** inspect logged exposure support and negative sampling after delivery checks.
Use independent judgments or justified randomized exploration.

**Check:** an unexposed item with no click is not an observed negative.

**Use when:** the previous policy selected the evaluation population.

## Failure and transfer

Immediate retraining changes the model while a stale serving preprocessor remains deployed, obscuring the fault.

What can shadow replay establish?

<details>
<summary>Check your answer</summary>

Prediction-path parity on replayed inputs, not how users respond to a changed live policy.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## The main cause families

**Objective mismatch:** a better proxy such as clicks or NDCG need not improve satisfaction or the declared user outcome.

**Evaluation bias:** inspect time/label leakage, repeated entities, biased negatives and unrepresentative data.

**Skew:** check feature definitions, freshness, defaults, preprocessing, versions and candidates.

**System effects:** latency, downstream load, inventory diversity and fallbacks can change product outcomes.

**Feedback:** exposure changes future labels; users, suppliers and adversaries may adapt.

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-adds"></span>
<span id="what-an-l6-answer-adds"></span>

## Distinguish evidence types

Counterfactual estimates require exposure support and justified assumptions.
Long-term holdbacks need valid assignment, enough duration and consideration of interference.
Neither is a universal substitute for controlled product measurement.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Keep the investigation falsifiable

Compare permissioned sampled inputs and outputs without exposing sensitive data.
Predeclare what result changes the next action, then improve the evaluation that missed the fault.

## Common follow-ups

New-user regression needs slice-specific parity checks.
For noisy delayed outcomes, wait for mature cohorts and use the planned uncertainty method.
Keep running only within authorized risk limits and predeclared monitoring/stopping rules, not because offline quality improved.

*Related: [delayed and selective labels](/concepts/delayed-labels-selective-labels-feedback-loops/), [point-in-time correctness](/concepts/data-leakage-point-in-time-correctness/), [causal inference](/concepts/causal-inference-for-ml-decisions/), and [A/B testing for ML](/concepts/ab-testing-for-ml/).*

</div>
