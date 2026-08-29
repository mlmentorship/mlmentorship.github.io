---
title: "Decision thresholds, asymmetric costs, and abstention"
description: "Choose actions from calibrated probabilities, error costs, and capacity constraints. Use separate thresholds for automatic action, human review, and abstention."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["classification threshold", "cost-sensitive classification", "reject option", "abstention", "human review threshold"]
roles: ["Applied Scientist", "Machine Learning Engineer", "Safety and Evals"]
rounds: ["Evaluation", "Product", "ML system design"]
difficulty: "Intermediate"
priority: "Core"
prerequisites: ["calibration", "precision-recall-f1"]
---

## Summary

A classifier score becomes a decision only after a threshold maps it to an action. The right threshold depends on the cost of false positives and false negatives, the probability calibration, review capacity, and constraints such as recall or safety limits.

Many systems need more than one threshold. High-confidence cases can receive an automatic action, uncertain cases can go to human review, and low-confidence cases can be allowed or rejected. Abstention is a valid action when the cost of deciding exceeds the cost of delay or escalation.

## Bayes decision rule

Suppose $p=P(Y=1\mid x)$ is calibrated. Let $C_{FP}$ be the cost of a false positive and $C_{FN}$ the cost of a false negative.

Predict positive when its expected error cost is lower:

$$
C_{FP}(1-p) < C_{FN}p.
$$

Solving for $p$ gives the threshold

$$
t^* = \frac{C_{FP}}{C_{FP}+C_{FN}}.
$$

If false negatives cost nine times as much as false positives, then $t^*=0.1$. The system should act at a lower probability because missing a positive is expensive.

This formula assumes the probabilities and costs are correct for the deployment population. Real systems often add policy constraints and operational limits.

## Scores are not probabilities

A ranking score can order examples well without representing $P(Y=1\mid x)$. The Bayes threshold formula does not apply directly to an arbitrary score.

Options include:

- calibrate the score on representative held-out data;
- choose a threshold from a precision-recall or ROC curve;
- optimize the measured business or safety objective directly;
- treat the threshold as a policy parameter and validate it online.

Recalibrate after a distribution shift, model update, or label-definition change.

## Constraints instead of costs

Teams often cannot assign credible dollar values to every error. Use an explicit constraint:

- maximize precision subject to recall at least 95%;
- minimize false negatives subject to 1,000 reviews per day;
- maximize accepted traffic subject to a safety-violation limit;
- minimize latency subject to a quality floor.

This turns threshold selection into an operating-point choice. Report the tradeoff curve and the chosen constraint.

## Review capacity

A review queue creates a hard capacity limit. If reviewers can inspect $K$ cases per day, choose the review region from the highest-value uncertain cases rather than a fixed score band by habit.

One policy may be:

```text
p >= 0.98          automatic block
0.60 <= p < 0.98  rank by expected review value
p < 0.60           allow
```

The queue can prioritize expected harm, model uncertainty, user impact, or information gain. Monitor backlog age and reviewer disagreement because the policy fails when the queue saturates.

## Abstention and selective prediction

A model with a reject option predicts only when confidence or estimated risk passes a rule. Two useful quantities are:

- **coverage:** fraction of examples that receive a model decision;
- **selective risk:** error rate on the decided examples.

Lower coverage can reduce selective risk if uncertainty estimates are useful. The system still needs a safe fallback for abstained examples.

Abstention can mean human review, a slower model, a rules system, a request for more information, or no action. Its cost belongs in the decision rule.

## Per-group thresholds

Different groups may have different base rates, error costs, or measurement quality. One global threshold can create unequal error rates.

Changing thresholds by group can improve one fairness criterion and harm another. It may also face legal or policy limits. State the target criterion, measurement uncertainty, and governance process. Do not present threshold adjustment as a complete fairness solution.

## Distribution shift

A threshold chosen on yesterday's data can fail when prevalence changes. Even with stable class-conditional score distributions, precision changes with the base rate.

Monitor:

- score and prevalence distributions;
- calibration by time and important slice;
- precision and recall at the operating point;
- action volume and review backlog;
- downstream harm and appeal outcomes;
- fallback use.

A stable AUC does not guarantee a stable operating point.

## Worked example

A fraud model estimates a 4% fraud probability. A false block costs $20$, while a missed fraud case costs $500$.

The cost-based threshold is

$$
t^*=\frac{20}{20+500}\approx 0.038.
$$

The expected-cost rule would block at 4%. If automatic blocking has legal or customer constraints, the same score may instead enter review. The final action depends on both expected cost and policy.

## In an interview

Use this order:

1. Define the available actions and error costs.
2. Ask whether the score is calibrated.
3. Derive or choose an operating point.
4. Add review capacity and abstention.
5. Check important slices and policy constraints.
6. Monitor calibration, action volume, and outcomes after launch.

## Common mistakes

- Using 0.5 as a universal threshold.
- Optimizing AUC without choosing an operating point.
- Applying probability formulas to uncalibrated scores.
- Ignoring review capacity and queue delay.
- Treating abstention as failure instead of an action.
- Selecting a threshold once and never monitoring it.

## Practice next

Use this framework in [calibration](/concepts/calibration/), [class imbalance](/questions/class-imbalance/), [fraud-system design](/questions/design-fraud-detection/), and [content moderation](/questions/content-moderation/).
