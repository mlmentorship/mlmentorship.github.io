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
