---
title: "Domain adaptation"
description: "Transfer a model across related but shifted data distributions without assuming unlabeled target data makes the problem identifiable."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Why it matters

Training on one distribution and deploying on a related but shifted one is the normal case, not the exception: a fraud model meets new fraud, a medical model meets a new hospital's scanner, a speech model meets a new accent. Domain adaptation transfers a predictor from a source distribution $P_s(X,Y)$ to a target $P_t(X,Y)$ where the label space is usually the same but the inputs, prevalences, or input-label relationship have moved. The first job is to name which of those moved, because that decides whether the problem is even solvable from the data you have.

## Shift types

- **Covariate shift:** $P(X)$ changes while $P(Y\mid X)$ is stable.
- **Label shift:** $P(Y)$ changes while $P(X\mid Y)$ is stable.
- **Concept shift:** $P(Y\mid X)$ changes; unlabeled target data alone is generally not enough.

Naming the assumed shift determines which correction is defensible.

## Approaches

- Importance weighting under a covariate- or label-shift assumption
- Fine-tuning on a small labeled target set
- Feature alignment with a discrepancy or adversarial objective
- Self-training with confidence and calibration controls
- Domain-specific normalization or adapters
- Robust optimization across the environments you can observe

## Evaluation

Use a true target-domain holdout and report the slices that matter. Validate calibration, not just ranking or accuracy, and measure negative transfer: adaptation can help the aggregate while hurting a target subgroup or eroding source performance.

## In an interview

1. Define source, target, labels, and how much target supervision you have.
2. State the shift assumption.
3. Establish source-only and target-labeled baselines.
4. Pick the simplest method the evidence justifies.
5. Monitor drift and collect the labels that distinguish concept shift from covariate shift.

## Common confusions

- **"Align the feature distributions and the task transfers."** Alignment can mix classes or erase the predictive structure you needed.
- **"Unlabeled target data solves domain shift."** Not when the label relationship itself changed.
- **"Fine-tuning always helps."** A small, biased target set can cause negative transfer and calibration failure.

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [epistemic versus aleatoric uncertainty](/concepts/epistemic-vs-aleatoric-uncertainty/), and [calibration](/concepts/calibration/).*
