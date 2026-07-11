---
title: "Design an ablation study that tests the claimed mechanism"
description: "Separate a model improvement from extra compute, data, parameters, tuning, and implementation confounds."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> A new architecture beats the baseline by 3%. Design the ablations needed to support the claim that the proposed component causes the improvement.

Ablation is not “remove every component one at a time.” The goal is to create experiments where the proposed mechanism and the strongest alternative explanation predict different outcomes.

## Clarify the claim

Ask:

- Is the claim about quality, efficiency, convergence, robustness, or transfer?
- Did the new model use more parameters, FLOPs, data, tuning trials, or training time?
- Is the component expected to matter in a specific regime?
- What variance exists across seeds and datasets?
- What is the strongest plausible alternative explanation?

## A strong ablation sequence

1. **Reproduce the result** across seeds with confidence intervals.
2. **Match resources:** parameters, training tokens, compute, wall time, and tuning budget where relevant.
3. **Remove the component** while holding the rest of the implementation fixed.
4. **Replace with a simple control** that matches capacity or compute without the claimed mechanism.
5. **Vary the mechanism strength** and test whether outcomes change as predicted.
6. **Test boundary regimes** where the claim predicts larger or no benefit.
7. **Measure mediators**—not only the final metric—if the mechanism predicts observable internal behavior.
8. **Replicate across tasks or datasets** only as broadly as the claim requires.

## What an L4 answer sounds like

> “Remove each layer and see how accuracy changes.”

This is a start, but it may compare models with different capacity and does not test the causal story.

## What an L5 answer adds

An L5 candidate controls compute and tuning, repeats seeds, and designs a simple matched baseline. They state which result would falsify the mechanism.

## What an L6 answer adds

An L6 candidate narrows the claim before expanding experiments:

- Which ablation changes the scientific conclusion rather than the paper table?
- Are benchmark and implementation choices selected after seeing results?
- Does the component merely make optimization easier at one scale?
- Would a cheaper intervention produce the same mediator?
- Is the claimed mechanism identifiable from these observations at all?

They prioritize high-information experiments instead of a combinatorial grid.

## Strong-hire signals

- You separate final performance from mechanism evidence.
- You match compute, capacity, and tuning opportunity.
- You state a falsifying result.
- You use predicted boundary conditions.
- You discuss variance and multiple comparisons.

## Down-leveling tells

- One seed.
- Comparing models with unequal training budgets.
- Calling feature importance an ablation without a causal claim.
- Reporting only the best configuration.
- Running every combination without prioritization.

## Likely follow-ups

- What if removing the component changes optimization stability?
- How do you match compute when architectures have different utilization?
- What mediator would support the proposed mechanism?
- How many seeds are enough?
- What result would make you reject the claim despite a positive average gain?

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [bias and variance of estimators](/concepts/bias-variance-of-estimators/), and [critique an ML paper](/questions/critique-ml-paper/).*
