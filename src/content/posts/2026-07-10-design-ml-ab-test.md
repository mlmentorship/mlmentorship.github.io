---
title: "Design an A/B test for a new ML model"
description: "Turn an offline model gain into a valid product decision: unit, exposure, power, guardrails, interference, and a ship rule."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> A new ranking model improves offline NDCG by 6%. Design the online experiment and decide what result would justify launch.

The interviewer is not asking for a list of statistical terms. They are testing whether you can turn uncertain evidence into a product decision without harming users.

## Start with the decision

Clarify before choosing metrics:

- What product behavior should improve?
- What is the minimum effect worth the engineering and serving cost?
- What user harms must not regress?
- Is launch reversible, staged, or permanent?
- Does the model change latency, content supply, or downstream systems?

A test without a decision threshold produces a p-value, not a decision.

## A strong structure

1. **Hypothesis:** what mechanism connects the model change to user value?
2. **Unit:** user, account, device, query, session, or geographic cluster?
3. **Assignment and exposure:** when is treatment assigned, and when is the model actually seen?
4. **Metrics:** one primary outcome, a small set of guardrails, and diagnostic slices.
5. **Power and duration:** minimum detectable effect, variance, traffic, seasonality, and novelty.
6. **Validity threats:** interference, carryover, sample-ratio mismatch, instrumentation, peeking, and multiple comparisons.
7. **Decision rule:** ship, iterate, stop, or expand only to a safer segment.

## What an L4 answer sounds like

> “Randomly split traffic 50/50, compare click-through rate, and launch if the result is statistically significant.”

This knows the basic experiment shape but leaves the product decision, guardrails, exposure, and validity threats unspecified.

## What an L5 answer adds

An L5 answer defines:

- User-level randomization for a persistent personalized experience
- A primary metric tied to the product objective—not whichever metric moves
- Guardrails for latency, complaints, retention, diversity, safety, and cost
- Power based on a minimum worthwhile effect
- An exposure analysis so assignment dilution does not hide impact
- A launch rule that includes practical significance and uncertainty

## What an L6 answer adds

An L6 answer challenges whether a standard user-level A/B test identifies the right effect:

- Marketplace or social interference may require cluster randomization
- Ranking changes can alter future training data, creating long-term feedback effects
- Novelty may reverse after the first week
- Heavy users can dominate aggregate metrics
- The treatment may improve engagement while degrading ecosystem health
- A holdback or staged ramp may be needed to measure slow effects

The strongest answer also says what evidence would make them **not** launch despite a positive primary metric.

## Strong-hire signals

- You distinguish assignment from exposure.
- You name the randomization unit and defend it.
- You choose a minimum worthwhile effect before seeing results.
- You include sample-ratio checks and instrumentation validation.
- You discuss heterogeneous effects and guardrails without fishing across dozens of slices.
- You state an operational rollout and rollback plan.

## Down-leveling tells

- “Statistically significant means ship.”
- CTR is selected because it is easy to move, not because it represents value.
- No treatment of latency or serving cost.
- No interference or carryover discussion.
- Repeatedly checking the result and stopping when $p < 0.05$.
- Inventing many metrics after seeing the data.

## Likely follow-ups

- What if assignment is user-level but users switch devices?
- What if only 30% of assigned users see the new model?
- What if CTR rises but seven-day retention falls?
- How would you test a model where users affect one another?
- What if the treatment changes the data used to retrain next month’s model?

*Related: [A/B testing for ML](/concepts/ab-testing-for-ml/), [How would you A/B test a chatbot?](/questions/ab-test-chatbot/), and [precision, recall, and F1](/concepts/precision-recall-f1/).*
