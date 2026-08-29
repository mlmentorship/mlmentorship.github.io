---
title: "Hypothesis testing and confidence intervals"
description: "Use uncertainty, effect size, and test assumptions to decide whether a measured model gain is likely to be real."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A hypothesis test asks whether observed data is hard to explain under a stated baseline. A confidence interval gives a range of effect sizes that are consistent with the data and the model assumptions.

## Why AI labs care

Model results contain noise. Training seeds, data order, sampled outputs, human ratings, and test examples can all change a score.

A result such as "the new model improved accuracy by 0.4 points" is incomplete. An interviewer will ask:

- How uncertain is the estimate?
- Were the same examples used for both models?
- How many model variants were tried?
- Is the gain large enough to matter?
- Which assumptions make the test valid?

## The basic setup

Suppose a baseline model has performance $m_A$ and a new model has performance $m_B$. The measured change is:

$$
\hat{\delta} = m_B - m_A.
$$

A common null hypothesis is:

$$
H_0: \delta = 0.
$$

The alternative hypothesis says that the true change is not zero, or that it is positive if the direction was chosen before the experiment.

A test statistic compares the measured change with its standard error:

$$
z = \frac{\hat{\delta}}{\operatorname{SE}(\hat{\delta})}.
$$

A large absolute value means the measured change is large relative to the noise estimate.

## What a p-value means

A p-value is the chance of seeing a result at least this extreme if the null hypothesis and test assumptions are true.

It is not:

- the chance that the null hypothesis is true;
- the chance that the result will reproduce;
- the size or value of the improvement;
- proof that the new method caused the change.

A small p-value can describe a tiny effect on a very large dataset. Report the effect size and its interval.

## Confidence intervals

A simple approximate confidence interval is:

$$
\hat{\delta} \pm z^* \operatorname{SE}(\hat{\delta}).
$$

For a 95% interval under a normal approximation, $z^*$ is about 1.96.

Example: a model gain is $0.8$ points with a 95% interval from $0.1$ to $1.5$ points. Zero is outside the interval. The data supports a positive effect under the assumptions. The interval also shows that the true gain may be too small to justify added cost.

## Use paired comparisons

Model A and Model B often score the same examples. Their errors are linked. Use the per-example difference instead of treating the two score sets as independent.

For accuracy, define one value per example:

$$
d_i = \mathbf{1}(B\text{ correct}) - \mathbf{1}(A\text{ correct}).
$$

Estimate the mean and uncertainty of $d_i$. A paired bootstrap or a test for paired binary outcomes is usually more efficient than an unpaired test.

For generation tasks, pair outputs by prompt. If humans compare two outputs, randomize their order and keep the prompt as the unit of analysis.

## Type I and Type II errors

- **Type I error:** report a gain when no true gain exists.
- **Type II error:** miss a real gain.
- **Power:** the chance of detecting an effect of a chosen size when it exists.

Low power does not make a negative result useful. Before running an expensive study, choose a minimum effect that would change the decision and estimate the required sample size.

## Multiple comparisons

If a team tries many models, prompts, benchmarks, and slices, one result may look strong by chance.

Good practice:

1. Choose the primary metric before looking at results.
2. Record all tested variants.
3. Treat broad slice analysis as exploration unless it was planned.
4. Confirm promising findings on fresh data.
5. Use a multiple-testing correction when many formal claims are made.

## Common assumptions

Check whether:

- examples are independent, or grouped by user, document, or task;
- the evaluation sample matches the target use;
- the metric has a stable variance;
- the test direction was chosen before seeing the result;
- failed or missing runs were included honestly;
- the test set was not used for model selection.

For repeated prompts from the same user or tasks from the same source, compute uncertainty at the group level.

## In an interview

Use this order:

1. State the decision and primary effect.
2. Name the unit of analysis.
3. Use a paired comparison when both models see the same examples.
4. Report effect size and confidence interval.
5. Discuss power, multiple comparisons, and important slices.
6. State what result would change the launch or research decision.

## Common mistakes

- Reporting only a p-value.
- Calling a non-significant result "no difference."
- Treating prompts from one user as independent samples.
- Picking the best seed and hiding the rest.
- Checking many metrics and reporting only the one that passed.
- Using statistical significance as the launch rule.

*Related: [expectation, variance, covariance, and correlation](/concepts/expectation-variance-covariance-correlation/), [bootstrap and resampling](/concepts/bootstrap-and-resampling/), [A/B testing for ML systems](/concepts/ab-testing-for-ml/), and [design an ablation study](/questions/design-ablation-study/).*