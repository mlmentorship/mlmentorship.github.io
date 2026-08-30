---
title: "Bootstrap and resampling"
description: "Estimate uncertainty by resampling the observed units. Preserve pairing and dependence, report the resampling unit, and know when the bootstrap can fail."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["nonparametric bootstrap", "paired bootstrap", "bootstrap confidence interval", "block bootstrap"]
roles: ["Applied Scientist", "Research Scientist", "Research Engineer"]
rounds: ["Statistics", "Evaluation", "Research"]
difficulty: "Intermediate"
priority: "Core"
prerequisites: ["expectation-variance-covariance-correlation", "hypothesis-testing-confidence-intervals"]
---

## Summary

The bootstrap estimates sampling uncertainty by repeatedly resampling observed units with replacement and recomputing a statistic. The distribution of those recomputed values approximates how the statistic would vary across new samples.

The method is useful when analytic standard errors are hard to derive. Its validity depends on the resampling unit and whether the observed sample represents the target population.

## Nonparametric bootstrap

Suppose the data contains $n$ observations $x_1,\ldots,x_n$ and the statistic is $T(x_1,\ldots,x_n)$.

For each bootstrap replicate:

1. Draw $n$ observations with replacement from the original sample.
2. Compute the statistic on that resample.
3. Store the result.

After $B$ replicates, the stored values approximate the sampling distribution of $T$.

A bootstrap sample repeats some observations and omits others. Each original observation has probability

$$
\left(1-\frac{1}{n}\right)^n \approx e^{-1}
$$

of being omitted. About 63.2% of distinct observations appear in a large bootstrap sample.

## Standard error and confidence intervals

The bootstrap standard error is the sample standard deviation of the replicate statistics:

$$
\widehat{\operatorname{SE}}_{\text{boot}}(T)
= \operatorname{sd}(T^{*(1)},\ldots,T^{*(B)}).
$$

A percentile interval uses empirical quantiles of the bootstrap distribution. A 95% interval takes the 2.5th and 97.5th percentiles.

Percentile intervals are simple but can be inaccurate for biased or highly skewed estimators. Basic, studentized, and bias-corrected accelerated intervals address different errors. In an interview, explain the simple interval first and then name the limitation.

## Paired model comparison

When two models score the same examples, preserve that pairing.

For each example $i$, compute the difference

$$
d_i = m_B(x_i)-m_A(x_i).
$$

Resample the examples, then average the selected differences. This paired bootstrap keeps the correlation between model outcomes. Resampling model A and model B separately throws away useful information and usually gives a wider or incorrect interval.

For generated outputs, resample prompts and keep both models' outputs for a prompt together. For repeated human ratings, decide whether the target population is prompts, raters, users, or a combination.

## Choose the resampling unit

The independent unit may be larger than one row.

| Data structure | Resampling unit |
| --- | --- |
| Independent examples | example |
| Many events per user | user |
| Documents split into passages | document |
| Queries with many candidates | query |
| Time series | time block |
| Multi-seed training study | seed or complete run |

If events from one user appear as independent bootstrap rows, the interval can become far too narrow. Resample the unit that could plausibly have been drawn again from the target population.

**Learning objective:** identify the independent sampling unit and keep every dependent observation attached to it when constructing a bootstrap replicate.

<!-- visual:bootstrap-resampling-unit -->
<figure class="learning-figure plot-panel" aria-labelledby="bootstrap-unit-title">
	<p class="visual-kicker">Resampling unit</p>
	<p class="visual-title" id="bootstrap-unit-title">If users are independent, draw users, not detached event rows.</p>
	<svg viewBox="0 0 360 408" role="img" aria-labelledby="bootstrap-unit-svg-title bootstrap-unit-svg-desc">
		<title id="bootstrap-unit-svg-title">Valid user-level bootstrap compared with invalid event-row bootstrap</title>
		<desc id="bootstrap-unit-svg-desc">The source dataset contains users U1, U2, and U3, each enclosing two event rows with paired model A and B outcomes. A valid bootstrap replicate draws U3, U1, and U3, carrying both events and both model outcomes whenever a user is selected. An invalid replicate samples detached event rows from across users and is crossed out because it breaks within-user dependence.</desc>
		<text class="viz-axis-label" x="12" y="18">SOURCE DATA · USERS ARE THE INDEPENDENT UNITS</text>
		<rect class="viz-plot-bg" x="10" y="28" width="340" height="92" rx="5"></rect>
		<rect class="viz-node viz-node--input" x="20" y="40" width="96" height="66" rx="4"></rect>
		<rect class="viz-node viz-node--input" x="132" y="40" width="96" height="66" rx="4"></rect>
		<rect class="viz-node viz-node--input" x="244" y="40" width="96" height="66" rx="4"></rect>
		<text class="viz-node-label" x="68" y="57">U1</text>
		<text class="viz-node-label" x="180" y="57">U2</text>
		<text class="viz-node-label" x="292" y="57">U3</text>
		<path class="viz-gridline" d="M29 64H107M141 64H219M253 64H331"></path>
		<text class="viz-node-value" x="68" y="79">event 1 · A + B</text>
		<text class="viz-node-value" x="68" y="95">event 2 · A + B</text>
		<text class="viz-node-value" x="180" y="79">event 1 · A + B</text>
		<text class="viz-node-value" x="180" y="95">event 2 · A + B</text>
		<text class="viz-node-value" x="292" y="79">event 1 · A + B</text>
		<text class="viz-node-value" x="292" y="95">event 2 · A + B</text>
		<text class="viz-axis-label" x="12" y="145">VALID REPLICATE · DRAW U3, U1, U3</text>
		<rect class="viz-plot-bg" x="10" y="155" width="340" height="112" rx="5"></rect>
		<rect class="viz-node viz-node--output" x="20" y="169" width="96" height="66" rx="4"></rect>
		<rect class="viz-node viz-node--output" x="132" y="169" width="96" height="66" rx="4"></rect>
		<rect class="viz-node viz-node--output" x="244" y="169" width="96" height="66" rx="4"></rect>
		<text class="viz-node-label" x="68" y="186">U3</text>
		<text class="viz-node-label" x="180" y="186">U1</text>
		<text class="viz-node-label" x="292" y="186">U3 again</text>
		<path class="viz-gridline" d="M29 193H107M141 193H219M253 193H331"></path>
		<text class="viz-node-value" x="68" y="208">event 1 · A + B</text>
		<text class="viz-node-value" x="68" y="224">event 2 · A + B</text>
		<text class="viz-node-value" x="180" y="208">event 1 · A + B</text>
		<text class="viz-node-value" x="180" y="224">event 2 · A + B</text>
		<text class="viz-node-value" x="292" y="208">event 1 · A + B</text>
		<text class="viz-node-value" x="292" y="224">event 2 · A + B</text>
		<path d="M42 246V255H318V246" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path>
		<text class="viz-callout" x="180" y="264" text-anchor="middle">each draw carries the whole user cluster</text>
		<text class="viz-axis-label" x="12" y="292">INVALID FOR THIS DATA · DRAW DETACHED EVENTS</text>
		<rect class="viz-plot-bg" x="10" y="302" width="340" height="94" rx="5"></rect>
		<rect class="viz-node viz-node--warning" x="20" y="317" width="96" height="42" rx="4"></rect>
		<rect class="viz-node viz-node--warning" x="132" y="317" width="96" height="42" rx="4"></rect>
		<rect class="viz-node viz-node--warning" x="244" y="317" width="96" height="42" rx="4"></rect>
		<text class="viz-node-label" x="68" y="335">U3 event 2</text>
		<text class="viz-node-value" x="68" y="350">A + B</text>
		<text class="viz-node-label" x="180" y="335">U1 event 1</text>
		<text class="viz-node-value" x="180" y="350">A + B</text>
		<text class="viz-node-label" x="292" y="335">U2 event 2</text>
		<text class="viz-node-value" x="292" y="350">A + B</text>
		<path d="M22 311L338 366M338 311L22 366" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5"></path>
		<text class="viz-callout" x="180" y="384" text-anchor="middle">breaks within-user dependence → uncertainty can look too small</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> first draw a user ID with replacement; then copy that user's entire box into the replicate. Repeating U3 is allowed and omitting U2 is allowed, but separating a user's events is not. The same rule preserves paired model outcomes: A and B stay attached to every selected event. Procedure checked against <a href="https://doi.org/10.1214/aos/1176344552">Efron's bootstrap paper</a>, <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html">SciPy's bootstrap documentation</a>, and <a href="https://doi.org/10.1017/CBO9780511802843"><cite>Bootstrap Methods and Their Application</cite></a>; the graphic is original.</figcaption>
</figure>

## Block and cluster bootstrap

A cluster bootstrap samples whole groups. It preserves dependence within each group and assumes groups are approximately independent.

A block bootstrap samples contiguous time windows. It preserves short-range temporal dependence. The block length must be long enough to retain relevant correlation, but short enough to provide enough distinct blocks.

For hierarchical data, a multi-stage bootstrap can sample users first and events within selected users second. State the population that each stage represents.

## Worked example

Two classifiers score the same 1,000 examples. Model B improves accuracy by 0.8 percentage points.

A paired bootstrap resamples example indices 10,000 times and recomputes the accuracy difference. Suppose the 95% percentile interval is $[-0.3, 1.9]$ points.

The estimate favors B, but the interval includes zero and meaningful losses. This result alone does not support a confident launch. More data or a clearer decision threshold is needed.

## When the bootstrap fails

The ordinary bootstrap can fail when:

- observations are dependent but rows are resampled independently;
- the sample is too small to represent rare outcomes;
- the estimator changes discontinuously near a boundary;
- extreme values dominate a heavy-tailed statistic;
- classes or slices absent from the sample matter in deployment;
- the data was selected by a policy that hides unobserved outcomes.

Resampling cannot create population coverage that the original sample lacks. It only reuses observed information.

## In an interview

Use this order:

1. Name the statistic and target population.
2. Choose the independent resampling unit.
3. Preserve model pairing.
4. Describe the resample-and-recompute loop.
5. Report the effect and interval.
6. Discuss dependence, rare events, and sample coverage.

A common follow-up asks how many replicates to use. A few thousand often estimates a standard error well; tail quantiles need more. Compute is rarely the main limit compared with choosing the right unit.

## Common mistakes

- Resampling rows when users or documents are the independent units.
- Breaking paired comparisons.
- Calling the bootstrap distribution a posterior distribution.
- Reporting an interval without the effect estimate.
- Treating resampling as a fix for biased data.
- Ignoring seed variation in model training.

## Practice next

Apply paired resampling in [hypothesis testing and confidence intervals](/concepts/hypothesis-testing-confidence-intervals/), [reproducible model comparison](/concepts/reproducibility-fair-model-comparison/), and [paper critique](/questions/critique-ml-paper/).
