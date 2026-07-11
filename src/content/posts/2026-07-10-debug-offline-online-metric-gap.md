---
title: "Your offline metric improved but the online metric got worse. Debug it."
description: "A senior experimentation question about objective mismatch, leakage, serving skew, feedback loops, and disciplined diagnosis."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> A model improves the main offline metric by 8%, but the online primary metric declines by 2%. What do you do?

The weak response generates explanations. The strong response orders measurements so each step eliminates a class of causes.

## First: verify the observation

Before explaining the gap:

1. Validate assignment, exposure, sample ratio, and metric instrumentation.
2. Confirm the online change is outside normal variance and not a novelty or seasonality artifact.
3. Check latency, error rate, fallback rate, and treatment delivery.
4. Reproduce the production feature and prediction path offline.

If the treatment was not delivered as intended, this is not yet a model-quality mystery.

## The main cause families

### Objective mismatch

The offline label or metric is only a proxy. Better NDCG, AUC, or accuracy may not improve the user outcome. The model may optimize clicks while reducing satisfaction, or improve average relevance while hurting high-value slices.

### Data leakage or evaluation bias

Time leakage, label leakage, repeated entities, biased negatives, or an unrepresentative test set can inflate offline results.

### Training–serving skew

Feature definitions, freshness, defaults, preprocessing, model version, or candidate sets differ in production.

### System effects

The model may increase latency, reduce inventory diversity, overload a downstream service, or trigger more fallbacks.

### Feedback and equilibrium effects

A ranking or recommendation model changes what users see and therefore changes future labels. Content suppliers, fraudsters, or other users may adapt.

## What an L4 answer sounds like

> “I would check data drift and retrain the model.”

Drift is one possibility, but retraining before validating experiment and serving integrity can make the diagnosis harder.

## What an L5 answer adds

An L5 candidate builds a decision tree:

- Was treatment assigned and exposed correctly?
- Did production predictions match an offline replay?
- Which slices account for the aggregate regression?
- Is the offline metric aligned with the product mechanism?
- Did latency, fallbacks, or candidate coverage change?
- Can a small rollback or shadow comparison isolate the cause safely?

## What an L6 answer adds

An L6 candidate asks whether the offline evaluation system itself should change:

- Does the benchmark encode the old policy’s selection bias?
- Are labels missing for items the old system never exposed?
- Did optimizing the proxy create a predictable second-order effect?
- Should the team add counterfactual evaluation, randomized exploration, long-term holdbacks, or a new guardrail?
- What organizational process allowed an 8% offline gain to bypass this risk?

## Strong-hire signals

- You verify the experiment before debugging the model.
- You distinguish model quality from system delivery.
- You use slice and counterfactual analysis to test hypotheses.
- You propose rollback or containment before a broad investigation.
- You improve the eval pipeline after identifying the mismatch.

## Down-leveling tells

- Immediately retraining on more data.
- Listing ten possible causes with no diagnostic order.
- Assuming the online metric is wrong because the offline metric improved.
- Ignoring latency, fallback, or candidate-generation changes.
- Looking only at averages.

## Likely follow-ups

- How do you compare online and offline predictions safely?
- What if the regression appears only for new users?
- What if the online primary metric is noisy and delayed?
- How would selection bias enter a recommender’s offline dataset?
- When would you keep the model running despite the initial regression?

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [A/B testing for ML](/concepts/ab-testing-for-ml/), and [evaluate a search ranker](/questions/evaluate-search-ranker/).*
