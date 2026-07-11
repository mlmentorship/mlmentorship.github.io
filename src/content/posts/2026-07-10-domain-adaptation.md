---
title: "Domain adaptation"
description: "Transfer a model across related but shifted data distributions without assuming unlabeled target data makes the problem identifiable."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Definition

Domain adaptation transfers a predictor from source distribution $P_s(X,Y)$ to related target distribution $P_t(X,Y)$. The label space usually stays the same, but inputs, prevalences, or conditional relationships shift.

## Shift types

- **Covariate shift:** $P(X)$ changes while $P(Y\mid X)$ is stable.
- **Label shift:** $P(Y)$ changes while $P(X\mid Y)$ is stable.
- **Concept shift:** $P(Y\mid X)$ changes; unlabeled target data alone is generally insufficient.

Naming the assumed shift determines which correction is defensible.

## Approaches

- Importance weighting under covariate or label-shift assumptions
- Fine-tuning with a small labeled target set
- Feature alignment using discrepancy or adversarial objectives
- Self-training with confidence and calibration controls
- Domain-specific normalization or adapters
- Robust optimization across observed environments

## Evaluation

Use a true target-domain holdout and report important slices. Validate calibration, not only ranking or accuracy. Measure negative transfer: adaptation can hurt target subgroups or source performance.

## Interview answer

1. Define source, target, labels, and available target supervision.
2. State the shift assumption.
3. Establish source-only and target-labeled baselines.
4. Choose the simplest method justified by available evidence.
5. Monitor drift and collect labels that distinguish concept shift from covariate shift.

## Common confusions

- **“Align feature distributions and the task transfers.”** Alignment can mix classes or erase predictive structure.
- **“Unlabeled target data solves domain shift.”** Not when the label relationship changes.
- **“Fine-tuning always helps.”** Small biased target sets can create negative transfer and calibration failure.

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [epistemic versus aleatoric uncertainty](/concepts/epistemic-vs-aleatoric-uncertainty/), and [calibration](/concepts/calibration/).*
