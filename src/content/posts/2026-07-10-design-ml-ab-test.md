---
title: "Design an A/B test for a new ML model"
description: "Turn an offline model gain into a valid product decision: unit, exposure, power, guardrails, interference, and a ship rule."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> A new ranking model improves offline NDCG by 6%. Design the online experiment and decide what result would justify launch.

Design an online test that turns a ranking model's offline improvement into a defensible product launch decision.

## Problem

**Context.** A proposed ranking model improves offline NDCG by 6%, but its effect on user outcomes has not been established.

**Task.** Choose randomization and analysis units, a primary outcome, guardrails, a minimum worthwhile effect, and an explicit release rule.
Specify power assumptions, exposure logging, ramp stages, and how validity failures or inconclusive results will be handled.

**Constraints.** Predeclare thresholds and stopping rules, account for interference and repeated observations, and stop unacceptable harm.
Offline relevance and statistical significance alone do not establish a practically valuable or safe launch.

**Learning objective:** Apply validity, practical-effect, and guardrail gates in order so a statistically significant result is not mistaken for a ship decision.

<!-- visual:ml-ab-test-launch-gates -->
<figure class="learning-figure editorial-illustration" aria-labelledby="ml-ab-test-launch-gates-edition-title"><p class="visual-title" id="ml-ab-test-launch-gates-edition-title">A positive effect can still be too small or unsafe to launch</p><div class="visual-argument" data-visual-argument="ml-ab-test-launch-gates" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="Significance, practical value, and safety are separate conditions. One candidate has too little practical gain; another fails safety. Neither passes all launch gates."><title>Significance, practical value, and safety are separate conditions</title><desc>One candidate has too little practical gain; another fails safety. Neither passes all launch gates.</desc><text x="105.13333333333333" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="105.13333333333333" dy="0">nonzero</tspan></text><text x="199.08" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="199.08" dy="0">worth it</tspan></text><text x="293.02666666666664" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="293.02666666666664" dy="0">safe</tspan></text><text x="46.16" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">case</tspan><tspan x="46.16" dy="18">A</tspan></text><g data-proof-status="present"><rect data-proof-cell="0,0" x="58.16" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M95.63333333333333 54l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="102.63333333333333" y="82" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="102.63333333333333" dy="0">yes</tspan></text><path d="M63.16 89L142.10666666666668 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="absent"><rect data-proof-cell="0,1" x="152.10666666666668" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="196.58" cy="54" r="9" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M191.58 54L201.58 54" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="196.58" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="196.58" dy="0">no</tspan></text><path data-proof-warning="0,1" d="M156.10666666666668 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="present"><rect data-proof-cell="0,2" x="246.05333333333334" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M283.5266666666667 54l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="290.5266666666667" y="82" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290.5266666666667" dy="0">yes</tspan></text><path d="M251.05333333333334 89L330 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="46.16" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">case</tspan><tspan x="46.16" dy="18">B</tspan></text><g data-proof-status="present"><rect data-proof-cell="1,0" x="58.16" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M95.63333333333333 118l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="102.63333333333333" y="146" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="102.63333333333333" dy="0">yes</tspan></text><path d="M63.16 153L142.10666666666668 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="present"><rect data-proof-cell="1,1" x="152.10666666666668" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M189.58 118l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="196.58" y="146" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="196.58" dy="0">yes</tspan></text><path d="M157.10666666666668 153L236.05333333333334 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="absent"><rect data-proof-cell="1,2" x="246.05333333333334" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="290.5266666666667" cy="118" r="9" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M285.5266666666667 118L295.5266666666667 118" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="290.5266666666667" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="290.5266666666667" dy="0">no</tspan></text><path data-proof-warning="1,2" d="M250.05333333333334 108v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 410" role="img" aria-labelledby="ml-ab-test-launch-gates-edition-svg-title ml-ab-test-launch-gates-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="ml-ab-test-launch-gates-edition-svg-title">Statistically positive, worthwhile, and safe are three different tests</title><desc id="ml-ab-test-launch-gates-edition-svg-desc">Under an illustrative precommitted one-point worthwhile threshold, interval 0.2 to 0.8 is positive but insufficient. Interval 1.1 to 1.8 is worthwhile but its two-percent error rate exceeds a one-percent guardrail. Both hold. If assignment validity fails, neither effect is interpretable.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative precommitted worthwhile</tspan><tspan x="180" dy="19">effect: +1 point</tspan></text><text x="18" y="72" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Significant, but too small</tspan></text><path d="M70 117L310 117" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M190 90L190 134" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M94 110L166 110" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle cx="130" cy="110" r="4" style="fill:var(--viz-input-stroke)"></circle><text x="70" y="152" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="70" dy="0">0</tspan></text><text x="190" y="152" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="190" dy="0">+1</tspan></text><text x="310" y="152" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="310" dy="0">+2 points</tspan></text><text x="18" y="210" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Worthwhile, but unsafe</tspan></text><path d="M70 255L310 255" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M190 228L190 272" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M202 248L286 248" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle cx="238" cy="248" r="4" style="fill:var(--viz-input-stroke)"></circle><text x="70" y="290" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="70" dy="0">0</tspan></text><text x="190" y="290" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="190" dy="0">+1</tspan></text><text x="310" y="290" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="310" dy="0">+2 points</tspan></text><text x="180" y="332" style="font-size:13px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">Second case: errors 2% &gt; 1% guardrail</tspan></text><text x="180" y="374" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Both HOLD. Invalid assignment? Interpret</tspan><tspan x="180" dy="19">neither.</tspan></text></svg><div class="visual-derivation-caption">A positive interval can miss the worthwhile threshold. A worthwhile effect can fail safety. If assignment is invalid, do not interpret either effect.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ml-ab-test-launch-gates"><strong>Check:</strong> Why can a significant primary result still fail the launch decision?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ml-ab-test-launch-gates">One candidate has too little practical gain; another fails safety. Neither passes all launch gates.</span></figcaption></figure>

## Start with the decision

Clarify before choosing metrics:

- What product behavior should improve, and through what mechanism?
- What is the minimum effect worth the engineering and serving cost?
- What user harms must not regress?
- Is launch reversible, staged, or permanent?
- Does the model change latency, content supply, or downstream systems?

## A test design that survives review

1. **Hypothesis:** what mechanism connects the model change to user value?
2. **Unit:** user, account, device, query, session, or geographic cluster?
3. **Assignment vs exposure:** when is treatment assigned, and when is the model actually seen?
4. **Metrics:** one primary outcome, a small set of guardrails, and diagnostic slices.
5. **Power and duration:** minimum detectable effect, variance, traffic, seasonality, and novelty.
6. **Validity threats:** interference, carryover, sample-ratio mismatch, instrumentation, peeking, and multiple comparisons.
7. **Decision rule:** ship, iterate, stop, or expand only to a safer segment.

## What an L4 answer sounds like

> "Randomly split traffic 50/50, compare click-through rate, and launch if the result is statistically significant."

It knows the shape of an experiment but leaves the decision, guardrails, exposure, and validity threats unspecified.

## What an L5 answer adds

An L5 answer defines the randomization unit and ties the primary metric to the product objective. It adds guardrails for latency, complaints, retention, diversity, safety, and cost. Power uses a minimum worthwhile effect. Exposure analysis catches assignment dilution, and the launch rule weighs practical significance against uncertainty.

## What an L6 answer adds

An L6 answer asks whether a user-level A/B test identifies the right effect. Marketplace or social interference may require cluster randomization. Ranking changes can alter future training data, novelty can fade, and heavy users can dominate aggregates. A positive engagement result may still harm ecosystem health.

## Tells that get you a strong-hire vote

- You distinguish assignment from exposure.
- You name the randomization unit and defend it.
- You fix a minimum worthwhile effect before seeing results.
- You include sample-ratio checks and instrumentation validation.
- You discuss heterogeneous effects and guardrails without fishing across dozens of slices.
- You state an operational rollout and rollback plan.

## Tells that get you down-leveled

- "Statistically significant means ship."
- CTR chosen because it is easy to move, not because it represents value.
- No treatment of latency or serving cost.
- No interference or carryover discussion.
- Peeking repeatedly and stopping when $p < 0.05$.
- Inventing new metrics after seeing the data.

## Common follow-ups

- What if assignment is user-level but users switch devices?
- What if only 30% of assigned users see the new model?
- What if CTR rises but seven-day retention falls?
- How would you test a model where users affect one another?
- What if the treatment changes the data used to retrain next month's model?

*Related: [A/B testing for ML](/concepts/ab-testing-for-ml/), [How would you A/B test a chatbot?](/questions/ab-test-chatbot/), and [precision, recall, and F1](/concepts/precision-recall-f1/).*
