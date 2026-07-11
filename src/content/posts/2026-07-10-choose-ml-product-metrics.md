---
title: "Choose metrics for an ML product"
description: "Select a primary outcome, guardrails, diagnostics, and decision horizon without turning the dashboard into a metric zoo."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> You are launching an ML-powered assistant. Choose the metrics the team should use to decide whether it is successful.

The interviewer is testing product judgment, causal reasoning, and whether you know the difference between a model metric and a product outcome.

## Start with the product decision

Clarify:

- Who is the user and what job are they trying to complete?
- Is the product optimizing speed, quality, discovery, revenue, safety, or cost?
- What failure is unacceptable?
- Over what time horizon should value appear?
- Which behavior can the model influence directly?

## A useful metric hierarchy

### One primary outcome

Choose the closest measurable outcome to user value that can change within the decision horizon. Examples include successful task completion, retained weekly use, qualified resolution, or incremental conversion—not raw model accuracy by default.

### Guardrails

Protect against predictable harm:

- User reports, unsafe outputs, or severe errors
- Latency and availability
- Cost per successful task
- Diversity or ecosystem concentration
- Retention or downstream quality

### Diagnostic metrics

Use these to explain the primary outcome, not to declare victory:

- Acceptance, edit, retry, abandonment, or escalation rate
- Retrieval coverage and citation support
- Model calibration and slice quality
- Tool-call and fallback behavior

### Offline model metrics

They support iteration and regression testing but do not replace the product outcome. State the expected mechanism linking offline change to online value.

## What an L4 answer sounds like

> “Accuracy, latency, user engagement, and cost.”

The list is reasonable but does not identify the decision, metric hierarchy, or trade-offs.

## What an L5 answer adds

An L5 candidate chooses one primary metric, explains why it represents user value, adds a small set of guardrails, and defines useful slices. They discuss instrumentation and what result would block launch.

## What an L6 answer adds

An L6 candidate identifies metric gaming and second-order effects:

- More assistant use can mean value—or repeated failure.
- Acceptance can rise because users stop checking.
- Faster completion can reduce learning or trust.
- Average quality can hide catastrophic failures in a small segment.
- A team can optimize a metric that the model itself changes how to measure.

They also specify governance: metric owner, review cadence, thresholds, and when the organization should revisit the objective.

## Strong-hire signals

- One primary metric rather than a dashboard vote.
- Clear separation of outcome, guardrail, diagnostic, and offline metrics.
- Cost per successful outcome, not cost per request alone.
- Slice strategy tied to plausible harms.
- A practical ship/iterate/stop rule.

## Down-leveling tells

- “Engagement” without defining valuable engagement.
- Twenty co-primary metrics.
- Relying on thumbs-up rate without accounting for response or selection bias.
- Ignoring delayed effects and user adaptation.
- Treating benchmark accuracy as the product goal.

## Likely follow-ups

- What if task completion rises but retention falls?
- How would you measure quality when labels arrive weeks later?
- What metric would a team accidentally game?
- Which slices should be mandatory before launch?
- How would metrics differ for a healthcare assistant?

*Related: [expected calibration error](/concepts/expected-calibration-error/), [A/B testing for ML](/concepts/ab-testing-for-ml/), and [evaluate an LLM application](/questions/how-would-you-evaluate-an-llm-application/).*
