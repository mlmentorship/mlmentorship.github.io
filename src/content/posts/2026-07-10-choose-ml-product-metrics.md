---
title: "Choose metrics for an ML product"
description: "Select a primary outcome, guardrails, diagnostics, and decision horizon without turning the dashboard into a metric zoo."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> You are launching an ML-powered assistant. Choose the metrics the team should use to decide whether it is successful.

Pick one primary metric that tracks user value, wrap it in a few guardrails, and keep model metrics as diagnostics. The failure mode is a dashboard of twenty co-primary metrics that can't decide anything, or an "engagement" number that climbs while the product is failing users.

## Start with the product decision

Clarify:

- Who is the user, and what job are they trying to finish?
- Is the product optimizing speed, quality, discovery, revenue, safety, or cost?
- What failure is unacceptable?
- Over what horizon should value appear?
- Which behavior can the model influence directly?

## A metric hierarchy

**One primary outcome.** The closest measurable outcome to user value that can move within the decision horizon: successful task completion, retained weekly use, qualified resolution, or incremental conversion. Raw model accuracy is not the default.

**Guardrails.** Protect against predictable harm: user reports and unsafe outputs, latency and availability, cost per successful task, diversity or ecosystem concentration, and retention or downstream quality.

**Diagnostics.** Explain the primary outcome without declaring victory: acceptance, edit, retry, abandonment, and escalation rate; retrieval coverage and citation support; calibration and slice quality; tool-call and fallback behavior.

**Offline model metrics.** They support iteration and regression testing but do not replace the product outcome. State the mechanism you expect to link an offline change to online value.

<!-- visual:product-metric-decision-authority -->
<figure class="learning-figure" aria-labelledby="metric-authority-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="metric-authority-title">Which metrics decide whether to ship, and which only help you understand the result?</p>
	<div class="visual-grid--two" role="group" aria-label="Product metrics grouped by their authority in a launch decision">
		<section class="visual-panel">
			<h4>DECISION GATE · CAN AUTHORIZE OR VETO</h4>
			<p><strong>1 · Primary outcome: authorize</strong><br />Did successful task completion improve by the pre-set amount within the decision horizon?</p>
			<p><strong>2 · Guardrails: veto</strong><br />Did unsafe-output rate, p95 latency, and cost per successful task all remain within their limits?</p>
			<p><strong>Ship rule</strong><br />Eligible only when the primary outcome clears its bar <em>and</em> every guardrail holds.</p>
		</section>
		<section class="visual-panel">
			<h4>SUPPORTING EVIDENCE · CANNOT DECLARE VICTORY</h4>
			<p><strong>Diagnostics: explain</strong><br />Retry, edit, abandonment, and escalation rates show where the experience changed and which slices need investigation.</p>
			<p><strong>Offline model metrics: qualify</strong><br />Task-eval and slice scores catch regressions and justify an online test; they do not prove user value.</p>
			<p><strong>Use rule</strong><br />Use these signals to debug, iterate, or reject a candidate; do not use them to outvote a missed primary outcome.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> read the left panel as an AND gate: the primary outcome must improve and no guardrail may fail. Read the right panel as evidence about why, not as extra votes that can turn a failed launch decision into a win.</figcaption>
</figure>

## What an L4 answer sounds like

> "Accuracy, latency, user engagement, and cost."

A reasonable list, but it names no decision, no hierarchy, and no trade-offs.

## What an L5 answer adds

An L5 answer picks one primary metric, explains why it represents user value, adds a small set of guardrails, defines the slices that matter, and states what result would block launch.

## What an L6 answer adds

An L6 answer anticipates gaming and second-order effects: more assistant use can mean value or repeated failure; acceptance can rise because users stop checking; faster completion can erode learning or trust; average quality can hide catastrophic failures in one segment. It also names an owner, a review cadence, thresholds, and when the organization should revisit the objective.

## Tells that get you a strong-hire vote

- One primary metric, not a dashboard vote.
- Clean separation of outcome, guardrail, diagnostic, and offline metrics.
- Cost per successful outcome, not cost per request alone.
- A slice strategy tied to plausible harms.
- A practical ship/iterate/stop rule.

## Tells that get you down-leveled

- "Engagement" with no definition of valuable engagement.
- Twenty co-primary metrics.
- Trusting thumbs-up rate without accounting for response or selection bias.
- Ignoring delayed effects and user adaptation.
- Treating benchmark accuracy as the product goal.

## Common follow-ups

- What if task completion rises but retention falls?
- How would you measure quality when labels arrive weeks later?
- Which metric would a team accidentally game?
- Which slices should be mandatory before launch?
- How would the metrics differ for a healthcare assistant?

*Related: [expected calibration error](/concepts/expected-calibration-error/), [A/B testing for ML](/concepts/ab-testing-for-ml/), and [evaluate an LLM application](/questions/how-would-you-evaluate-an-llm-application/).*
