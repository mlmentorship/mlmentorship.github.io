---
title: "Your offline metric improved but the online metric got worse. Debug it."
description: "A senior experimentation question about objective mismatch, leakage, serving skew, feedback loops, and disciplined diagnosis."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> A model improves the main offline metric by 8%, but the online primary metric declines by 2%. What do you do?

Order your measurements so each step rules out a class of causes. The weak answer free-associates explanations; the strong answer verifies the experiment first, then the serving path, then model quality, and reaches for "drift" only after the cheaper causes are gone.

## First: verify the observation

Before explaining the gap:

1. Validate assignment, exposure, sample-ratio, and metric instrumentation.
2. Confirm the online change exceeds normal variance and is not novelty or seasonality.
3. Check latency, error rate, fallback rate, and whether the treatment was actually delivered.
4. Reproduce the production feature and prediction path offline.

If the treatment was not delivered as intended, this is not yet a model-quality mystery.

## The main cause families

**Objective mismatch.** The offline label or metric is a proxy. Better NDCG, AUC, or accuracy need not improve the user outcome: the model may lift clicks while cutting satisfaction, or raise average relevance while hurting high-value slices.

**Data leakage or evaluation bias.** Time leakage, label leakage, repeated entities, biased negatives, or an unrepresentative test set inflate offline results.

**Training-serving skew.** Feature definitions, freshness, defaults, preprocessing, model version, or candidate sets differ in production.

**System effects.** The model may add latency, cut inventory diversity, overload a downstream service, or trigger more fallbacks.

**Feedback and equilibrium effects.** A ranking model changes what users see and therefore changes future labels; suppliers, fraudsters, or users adapt.

## What an L4 answer sounds like

> "I would check data drift and retrain the model."

Drift is one candidate, but retraining before you validate experiment and serving integrity only makes the diagnosis harder.

## What an L5 answer adds

An L5 answer builds a decision tree: was treatment assigned and exposed correctly; do production predictions match an offline replay; which slices account for the aggregate regression; is the offline metric aligned with the product mechanism; did latency, fallbacks, or candidate coverage change; can a small rollback or shadow comparison isolate the cause safely.

## What an L6 answer adds

An L6 answer questions the offline evaluation system itself: does the benchmark encode the old policy's selection bias; are labels missing for items the old system never showed; did optimizing the proxy create a predictable second-order effect; should the team add counterfactual evaluation, randomized exploration, or long-term holdbacks; and what process let an 8% offline gain bypass this risk.

## Tells that get you a strong-hire vote

- You verify the experiment before debugging the model.
- You separate model quality from system delivery.
- You use slice and counterfactual analysis to test hypotheses.
- You propose rollback or containment before a broad investigation.
- You improve the eval pipeline once you find the mismatch.

## Tells that get you down-leveled

- Immediately retraining on more data.
- Listing ten possible causes with no diagnostic order.
- Assuming the online metric is wrong because the offline metric improved.
- Ignoring latency, fallback, or candidate-generation changes.
- Looking only at averages.

## Common follow-ups

- How do you compare online and offline predictions safely?
- What if the regression appears only for new users?
- What if the online primary metric is noisy and delayed?
- How would selection bias enter a recommender's offline dataset?
- When would you keep the model running despite the initial regression?

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [A/B testing for ML](/concepts/ab-testing-for-ml/), and [evaluate a search ranker](/questions/evaluate-search-ranker/).*
