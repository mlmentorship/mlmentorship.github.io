---
title: "Design an ablation study that tests the claimed mechanism"
description: "Separate a model improvement from extra compute, data, parameters, tuning, and implementation confounds."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> A new architecture beats the baseline by 3%. Design the ablations needed to support the claim that the proposed component causes the improvement.

Ablation is not "remove every component one at a time." The goal is experiments where the proposed mechanism and the strongest alternative explanation (more parameters, more compute, more tuning) predict different outcomes. Design for that contrast and you separate the science from the confounds.

## Clarify the claim

Ask:

- Is the claim about quality, efficiency, convergence, robustness, or transfer?
- Did the new model use more parameters, FLOPs, data, tuning trials, or training time?
- Is the component expected to matter only in a specific regime?
- How much variance is there across seeds and datasets?
- What is the strongest plausible alternative explanation?

## A strong ablation sequence

1. **Reproduce the result** across seeds with confidence intervals.
2. **Match resources:** parameters, training tokens, compute, wall time, and tuning budget.
3. **Remove the component** while holding the rest of the implementation fixed.
4. **Replace with a simple control** that matches capacity or compute without the claimed mechanism.
5. **Vary the mechanism strength** and test whether outcomes change as predicted.
6. **Test boundary regimes** where the claim predicts a larger benefit or none.
7. **Measure mediators**, not only the final metric, when the mechanism predicts observable internal behavior.
8. **Replicate across tasks or datasets** only as broadly as the claim requires.

## What an L4 answer sounds like

> "Remove each layer and see how accuracy changes."

A start, but it can compare models of different capacity and never tests the causal story.

## What an L5 answer adds

An L5 answer controls compute and tuning, repeats seeds, designs a simple matched baseline, and states which result would falsify the mechanism.

## What an L6 answer adds

An L6 answer narrows the claim before expanding the experiment set. It asks which ablation could change the scientific conclusion, whether choices were made after seeing results, and whether the component only helps optimization at one scale. It also tests cheaper interventions and whether the observations can identify the claimed mechanism.

## Tells that get you a strong-hire vote

- You separate final performance from mechanism evidence.
- You match compute, capacity, and tuning opportunity.
- You state a falsifying result.
- You use predicted boundary conditions.
- You account for variance and multiple comparisons.

## Tells that get you down-leveled

- One seed.
- Comparing models with unequal training budgets.
- Calling feature importance an ablation with no causal claim.
- Reporting only the best configuration.
- Running every combination without prioritization.

## Common follow-ups

- What if removing the component changes optimization stability?
- How do you match compute when architectures have different utilization?
- What mediator would support the proposed mechanism?
- How many seeds are enough?
- What result would make you reject the claim despite a positive average gain?

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [bias and variance of estimators](/concepts/bias-variance-of-estimators/), and [critique an ML paper](/questions/critique-ml-paper/).*
