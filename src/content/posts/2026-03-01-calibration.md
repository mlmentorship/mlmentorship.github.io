---
title: "Calibration: make probabilities match their event"
description: "Name the predicted event, check held-out frequencies, calculate temperature scaling, and separate calibration, ranking, accuracy, and probability-based actions."
date: "2026-03-01"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Calibration concerns a probability's meaning, not which class ranks first.
For binary event $Y=1$, calibration means $\mathbb E[Y\mid p]=p$ in the evaluated population.
This is different from calibrating predicted-label correctness, and weaker than calibrating every component of a multiclass distribution.

**Task:** ten untouched hypothetical binary cases all have positive-event probability $p=0.8$; six labels are one.
Evaluate temperature $T=2$, fitted on separate calibration data.
Calculate the new probability, reliability gap, Brier score, and decisions at thresholds $0.5$ and $0.75$.

Small sample frequencies have uncertainty; this fixture cannot establish population-wide improvement.

## Measuring calibration

Use representative evaluation cases whose labels are mature and whose event definition is unchanged.

1. Bin positive-event probabilities; report bin counts and mean probabilities.
2. Count positive labels, not generic classification correctness, for each bin.
3. Plot mean probability against positive fraction; the diagonal marks agreement.
4. Report uncertainty and complementary proper scores such as Brier loss and log loss.

Event-frequency ECE averages absolute bin gaps using bin-frequency weights.
It depends on binning and can conceal within-bin or subgroup errors; it does not measure discrimination by itself.
For top-label calibration instead, define confidence as the selected class's probability and outcomes as prediction correctness.

<!-- visual:calibration-reliability-gap -->
<figure class="learning-figure plot-panel" aria-labelledby="calibration-visual-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="calibration-visual-title">Compare positive-event frequency with its predicted probability.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 310" role="img" aria-labelledby="calibration-svg-title calibration-svg-desc">
			<title id="calibration-svg-title">Illustrative reliability diagram for positive-event probability</title>
			<desc id="calibration-svg-desc">Horizontal values are mean predicted positive-event probabilities; vertical values are observed positive fractions. Five invented bins contain 100 cases each, with positive counts 12, 28, 43, 60, and 72 at probabilities 0.2, 0.4, 0.6, 0.8, and 0.9. All points lie below the diagonal: the event probability is overestimated. The 0.8 bin has a 20-percentage-point gap. These are illustrations, not measurements.</desc>
			<rect class="viz-plot-bg" x="54" y="28" width="272" height="222" rx="3"></rect>
			<path class="viz-gridline" d="M54 194.5H326 M54 139H326 M54 83.5H326 M122 28V250 M190 28V250 M258 28V250"></path>
			<path class="viz-axis" d="M54 28V250H326"></path>
			<path class="viz-baseline" d="M54 250L326 28"></path>
			<text class="viz-axis-label" x="110" y="173" transform="rotate(-39 110 173)">perfect calibration</text>
			<text class="viz-label" x="77" y="64">EVENT UNDERESTIMATED</text>
			<text class="viz-label" x="217" y="220">EVENT OVERESTIMATED</text>
			<path class="viz-pr-curve" d="M108.4 223.4 L162.8 187.8 L217.2 154.5 L271.6 116.8 L298.8 90.2"></path>
			<circle class="viz-operating-point" cx="108.4" cy="223.4" r="4"></circle>
			<circle class="viz-operating-point" cx="162.8" cy="187.8" r="4"></circle>
			<circle class="viz-operating-point" cx="217.2" cy="154.5" r="4"></circle>
			<circle class="viz-operating-point" cx="271.6" cy="116.8" r="5"></circle>
			<circle class="viz-operating-point" cx="298.8" cy="90.2" r="4"></circle>
			<path class="viz-operating-guide" d="M271.6 72.4V116.8"></path>
			<path class="viz-operating-guide" d="M266.6 72.4H276.6 M266.6 116.8H276.6"></path>
			<text class="viz-callout" x="263" y="92" text-anchor="end">20-point gap</text>
			<text class="viz-callout" x="263" y="107" text-anchor="end">80% vs 60%</text>
			<text class="viz-callout" x="152" y="211">illustrative bins</text>
			<text class="viz-label" x="50" y="268">0</text>
			<text class="viz-label" x="186" y="268">0.5</text>
			<text class="viz-label" x="320" y="268">1</text>
			<text class="viz-label" x="36" y="254">0</text>
			<text class="viz-label" x="30" y="143">0.5</text>
			<text class="viz-label" x="36" y="33">1</text>
			<text class="viz-axis-label" x="190" y="294" text-anchor="middle">mean predicted probability</text>
			<text class="viz-axis-label" transform="translate(14 194) rotate(-90)">observed positive fraction</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> the diagonal marks probability-frequency agreement. Each invented bin has 100 cases, with positive counts 12, 28, 43, 60, and 72. All five points overestimate this event, not necessarily predicted-label correctness. At probability 0.8 versus positive frequency 0.6, the gap is 20 points.</figcaption>
</figure>

<span id="why-neural-networks-are-overconfident"></span>
## Why confidence can be wrong

Finite samples, model misspecification, fitting choices, and deployment shift can break probability-frequency agreement.
[Guo et al.](https://arxiv.org/abs/1706.04599) studied particular neural classification settings, not a universal ordering of every model family.

Cross-entropy is a proper probability objective; minimizing empirical loss is not a guarantee of correct population probabilities.

## How to fix calibration

### Temperature scaling

Use $p_T=\sigma(z/T)$ for binary logits, or shared-scalar $\operatorname{softmax}(z/T)$ for multiclass scores, with $T>0$.
Fit the temperature on calibration data and evaluate the frozen map on independent cases.

Here $z=\log(0.8/0.2)=\log4$.
Dividing by two gives $\log2$, hence $p_T=2/3$.
The observed positive fraction remains $6/10=0.6$.

| Check | Original | After separate $T=2$ fit |
| --- | --- | --- |
| Absolute reliability gap | $\lvert 0.8-0.6\rvert=0.2$ | $\lvert 2/3-0.6\rvert=1/15$ |
| Brier loss | $(6(0.2)^2+4(0.8)^2)/10=0.28$ | $(6(1/3)^2+4(2/3)^2)/10=11/45$ |
| Positive decisions at $0.5$ | Ten | Ten |
| Positive actions at $0.75$ | Ten | Zero |

Argmax accuracy remains six of ten.
Shared positive temperature preserves class ordering, not every probability-based action.

### Platt scaling

Fit $\sigma(af(x)+b)$ to binary calibration labels.
Its slope and intercept can change a threshold; an unconstrained negative slope can reverse ordering.

### Isotonic regression

**New task:** replace a one-parameter map when nonlinear distortion persists and calibration data support a more flexible map.
In a separate calibration fixture, three equally sized groups have scores $[0.2,0.4,0.6]$ and event rates $[0.3,0.7,0.5]$.
Each group has 20 cases, with positive counts $[6,14,10]$.
An increasing least-squares isotonic fit pools the last two rates to $24/40=0.6$, giving $[0.3,0.6,0.6]$.
**Check:** previously distinct scores tie; both now exceed a $0.5$ threshold.
**Use when:** scalar temperature leaves systematic errors; independently evaluate the frozen map and its plateaus.

### Change the population

**New task:** deployment prevalence changes while score ordering stays stable.
Collect mature representative labels and retest the same event by relevant slice.
**Check:** a probability-$0.8$ bucket with five positive labels among ten now has empirical gap $0.3$.
**Use when:** traffic, labeling, or policy changes; ordering stability alone does not establish calibration.

**Failure:** always predicting a population's base rate can be calibrated while distinguishing no cases.

**Reconstruct:** does lowering generation temperature calibrate factual confidence?

<details><summary>Answer</summary>

No. Decoding changes token sampling.
Calibration requires a defined event and held-out frequency evidence, not merely more deterministic output.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

### Label smoothing during training

For $K$ classes, use $(1-\epsilon)y+\epsilon/K$ under a declared smoothing convention.
This changes training targets, not a post-hoc calibration fit; evaluate probability quality rather than assuming improvement.

### Ensembling

Averaging probabilities changes the predictive distribution and inference cost.
Neither lower error nor better calibration is guaranteed for every ensemble.

### Bayesian methods

Posterior prediction, variational approximations, and Monte Carlo dropout represent different uncertainty procedures.
Model assumptions and approximations still need checking; Bayesian construction alone does not prove empirical calibration under shift.

## Calibration for LLMs

Next-token probability describes a token event conditional on context, not directly the truth of an entire answer.
Self-consistency measures sampled agreement; confidently repeated errors can agree.
Verbalized confidence is another model output, requiring a task-specific event definition and representative labeled evaluation.
Do not rank these signals universally or treat post-training methods as automatically destroying every notion of calibration.

## What an interviewer expects you to say

State the event, evaluated population, fitting/evaluation split, sample counts, and downstream decision rule.

## Common confusions

Accuracy, discrimination, calibration, and decision cost answer different questions.
Predictive intervals also have coverage notions; proper distributional scores such as CRPS measure more than calibration alone.
Top-label calibration does not certify every class probability or subgroup.

## Why interviewers ask

For a frozen argmax-only action, shared positive temperature leaves that action unchanged.
Abstention, thresholds, and expected-cost policies can depend on the changed probabilities.

---

*Related: [decision thresholds, asymmetric costs, and abstention](/concepts/decision-thresholds-asymmetric-costs-abstention/), [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), and [How do you handle hallucinations in production?](/questions/handle-hallucinations-in-production/).*

</div>
