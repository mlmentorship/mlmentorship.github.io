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

The bootstrap recomputes a statistic after drawing observed sampling units with replacement.
Its empirical resampling distribution can approximate sampling uncertainty under appropriate sampling, dependence, and estimator conditions.
Choose the target population and independent unit first; more simulated replicates cannot repair missing population information.

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

For a mechanics fixture, independent users U1, U2, and U3 have paired model differences $-1,0,1$ respectively.
Each user has two perfectly correlated events with that same difference; the target is the equal-user mean, zero.
Draw U3, U1, U3 and carry both events and both model outcomes whenever a user is selected.
The replicate user mean is $(1-1+1)/3=1/3$.

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
	<figcaption><strong>Read it this way:</strong> draw a user and carry the entire cluster. U3 can repeat and U2 can be omitted. Keep both models' paired outcomes attached to every selected event. The original graphic follows <a href="https://doi.org/10.1214/aos/1176344552">Efron</a>, <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html">SciPy</a>, and <a href="https://doi.org/10.1017/CBO9780511802843"><cite>Bootstrap Methods and Their Application</cite></a>. <span class="visual-reader-check" data-reader-check="bootstrap-resampling-unit"><strong>Check:</strong> Which records must travel together when U3 is drawn twice?</span></figcaption>
</figure>

## Nonparametric bootstrap

With $n$ independent units and statistic $T$, draw $n$ unit indices with replacement and recompute $T$.
Repeat $B$ times while preserving the same estimand and within-unit structure.
An original unit is omitted with probability $(1-1/n)^n$, approaching $e^{-1}$.
The familiar $63.2\%$ inclusion figure is an asymptotic expectation, not an exact count in each replicate.

The three-user fixture has $3^3=27$ equally likely ordered draws.
Replicate sums $-3,-2,-1,0,1,2,3$ have counts $1,3,6,7,6,3,1$; divide each sum by three.
Their probability-weighted mean is zero and variance is $2/9$, giving standard deviation $\sqrt{2/9}\approx0.471405$.
Equivalently, the empirical user variance $2/3$ is divided by three independent empirical draws.
This exact conditional distribution illustrates mechanics; three users do not establish reliable interval coverage.

## Standard error and confidence intervals

For randomly simulated replicates, estimate bootstrap standard error by their sample standard deviation.
Exact enumeration instead uses the probability-weighted variance over all possible draws, as above.
A percentile interval uses empirical quantiles, conventionally $2.5\%$ and $97.5\%$ for a nominal $95\%$ interval.
Nominal coverage is not guaranteed; bias, skewness, dependence, and estimator irregularity matter.
Increasing $B$ reduces simulation error, not the original sampling uncertainty.

## Paired model comparison

For models scored on the same units, retain $d_i=m_B(x_i)-m_A(x_i)$ within every resampled unit.
Breaking pairing changes the resampling distribution; it need not always widen the interval.
For the fixture, incorrectly drawing six detached events gives mean variance $1/9$, half the user-bootstrap variance.
Decide whether prompts, raters, users, or training runs supply the independence relevant to the desired uncertainty.

### Change Cluster Sizes

**New task:** Give U1 and U2 one event each, but U3 eight events.

**Picture and change:** Retain user means for equal-user performance.
For event-weighted performance, divide summed event differences by selected event count.

**Check:** Equal-user effect remains zero; observed event-weighted effect is $(-1+0+8)/10=0.7$.
For draw U3, U1, U3, those estimators give $1/3$ versus $15/17$.

**Use when:** Unequal exposure changes whether the target represents a random user or a random event.

### Preserve Time Dependence

**New task:** Resample `[0,0,1,1]` with dependence inside length-two blocks.

**Picture and change:** Draw whole blocks `[0,0]` and `[1,1]`, retaining paired outcomes.

**Check:** Drawing both blocks gives mean $0.5$; drawing the first twice gives zero.

**Use when:** Relevant local dependence fits the declared block length and stationarity assumptions.

<!-- learning:core-end -->
<div data-learning-extension>

## Block and cluster bootstrap

A cluster bootstrap samples whole groups. It preserves dependence within each group and assumes groups are approximately independent.

A block bootstrap samples contiguous time windows. It preserves short-range temporal dependence. The block length must be long enough to retain relevant correlation, but short enough to provide enough distinct blocks.

For hierarchical data, a multi-stage bootstrap can sample users first and events within selected users second. State the population that each stage represents.

## Worked example

As a separate hypothetical decision illustration, two fixed classifiers differ by positive $0.8$ accuracy percentage points on one thousand examples.
Suppose a suitable paired analysis reports interval $[-0.3,1.9]$ points.
These illustrative endpoints are not computed from the three-user fixture or unspecified per-example outcomes.
The result includes possible losses; a launch still needs decision thresholds, guardrails, and an appropriate uncertainty argument.

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

Choose replicate count from the required simulation accuracy, particularly for tail quantiles, after establishing a defensible sampling unit.
Fixed-model example resampling does not by itself capture variation from retraining with different seeds or datasets.

## Common mistakes

- Resampling rows when users or documents are the independent units.
- Breaking paired comparisons.
- Calling the bootstrap distribution a posterior distribution.
- Reporting an interval without the effect estimate.
- Treating resampling as a fix for biased data.
- Ignoring seed variation in model training.

## Practice next

Apply paired resampling in [hypothesis testing and confidence intervals](/concepts/hypothesis-testing-confidence-intervals/), [reproducible model comparison](/concepts/reproducibility-fair-model-comparison/), and [paper critique](/questions/critique-ml-paper/).

</div>
