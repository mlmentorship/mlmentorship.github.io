---
title: "Critique an ML paper you have not seen before"
description: "A research-depth framework for claims, baselines, evidence, leakage, compute fairness, ablations, and generalization."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> You have ten minutes to read a paper abstract, method figure, and main result table. Critique the work and propose the next experiment.

The interviewer is not testing whether you can find faults quickly. They want fair, prioritized scientific judgment under incomplete information.

## A strong reading order

1. **Claim:** what does the paper say is new and true?
2. **Evidence:** which result supports each part of the claim?
3. **Comparison:** are baselines strong, current, and fairly resourced?
4. **Validity:** could leakage, selection, tuning, or variance explain the result?
5. **Mechanism:** do ablations isolate why it works?
6. **Scope:** where should the conclusion generalize, and where should it fail?
7. **Value:** is the gain practically meaningful relative to compute and complexity?
8. **Next experiment:** what single result would most change your belief?

## Be fair before being critical

Start by stating the strongest contribution in the authors’ own terms. Separate:

- A correct result with an overstated claim
- A useful engineering improvement without novel science
- A plausible mechanism with insufficient evidence
- A flawed evaluation that invalidates the headline

## What an L4 answer sounds like

> “The gain is small, there are not enough datasets, and they should compare to more baselines.”

These may be true, but they are generic and do not identify the highest-impact threat.

## What an L5 answer adds

An L5 candidate connects criticism to the claim:

- If the claim is efficiency, compare matched wall time and hardware utilization.
- If the claim is robustness, define the shift and uncertainty.
- If the claim is a mechanism, require a discriminating ablation.
- If the gain is small, compare it with seed variance and tuning budget.

They propose one feasible next experiment rather than a wish list.

## What an L6 answer adds

An L6 candidate evaluates strategic value:

- Is the benchmark saturated or misaligned with real use?
- Does the method alter the cost or reliability frontier?
- Which result is likely to survive scale?
- What organizational capability would be needed to reproduce it?
- Is the contribution a new primitive, a recipe, or a local optimization?

They update visibly when evidence contradicts their first impression.

## Strong-hire signals

- Critique is prioritized by impact on the central claim.
- Baseline and compute fairness are explicit.
- You distinguish absence of evidence from evidence of failure.
- You propose a falsifiable next experiment.
- You acknowledge a real strength before limitations.

## Down-leveling tells

- Reviewing reputation or venue rather than evidence.
- Demanding more datasets without explaining what they test.
- Treating a small gain as meaningless without uncertainty or cost context.
- Missing leakage or matched-compute issues.
- Producing ten criticisms and no decision.

## Likely follow-ups

- Would you invest a month reproducing this paper?
- Which claim would you narrow?
- What is the strongest alternative explanation?
- How would you test whether the result survives scale?
- What if the method is slower but easier to operate?

*Related: [design an ablation study](/questions/design-ablation-study/), [bias–variance of estimators](/concepts/bias-variance-of-estimators/), and [lessons from Marin 8B](/guides/lessons-from-marin-8b/).*
