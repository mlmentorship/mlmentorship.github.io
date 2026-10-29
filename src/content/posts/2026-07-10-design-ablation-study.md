---
title: "Design an ablation study that tests the claimed mechanism"
description: "Separate a model improvement from extra compute, data, parameters, tuning, and implementation confounds."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

Design an ablation that distinguishes a claimed model mechanism from plausible alternative explanations for an observed gain.

## Problem

**Context.** A hypothetical routing recipe improves accuracy over an original baseline, and its authors attribute the difference to content-dependent routing.

**Task.** Specify a control, matched resources, measurement plan, and result patterns that would support or weaken that explanation.
Explain whether the comparison tests the whole recipe or isolates the claimed mechanism.

**Constraints.** Control training budget, data, optimization, and evaluation where relevant.
Removing a component can also change capacity or compute, so a score drop alone does not isolate why it mattered.

An explicitly hypothetical router F scores 83% accuracy on distant-evidence tasks; original baseline A scores 80%.
The claim attributes the three-percentage-point gain to content-dependent routing.
Design a control and decide what evidence would support that explanation.

## Picture

Use the same router architecture with position-only input as control C.
Suppose C scores 82.8%, while F scores 83%.
Most of the headline improvement remains without content-dependent input.
These are teaching means, not measured research results or uncertainty estimates.

Before confirmation, predict at least two points for F−C on distant evidence.
Predict at most 0.5 on short evidence.
The control must retain usable capacity, not merely unused parameters.

| Regime | F | C | F−C |
| --- | --- | --- | --- |
| Distant evidence | $83\%$ | $82.8\%$ | $0.2$ points |
| Short evidence | $91\%$ | $90.9\%$ | $0.1$ points |

The difference of controlled gaps is $0.2-0.1=0.1$ points.
The displayed distant mean misses the predicted useful effect; control validity and uncertainty determine how strongly to narrow the claim.

<!-- visual:ablation-rival-hypotheses -->
<figure class="learning-figure plot-panel" aria-labelledby="ablation-rival-title">
	<p class="visual-kicker">Hypothetical controlled comparison</p>
	<p class="visual-title" id="ablation-rival-title">The matched-control gap is 0.2 points, not the three-point headline</p>
	<svg viewBox="0 0 360 530" role="img" aria-labelledby="ablation-rival-svg-title ablation-rival-svg-desc">
		<title id="ablation-rival-svg-title">Three-point headline improvement versus a 0.2-point controlled gap</title>
		<desc id="ablation-rival-svg-desc">Hypothetical distant-evidence accuracy is 80 percent for original baseline A, 82.8 for matched position-only control C and 83 for content router F. The axis is deliberately restricted to 80 through 84 percent, with x equal to 60 plus 64 times accuracy minus 80. The full versus original gain spans three points; the full versus control gap is only 0.2. A second regime has controlled gap 0.1, so the difference of gaps is 0.1. The predeclared prediction requires distant gain at least two points and short gain at most 0.5. Means alone do not provide uncertainty.</desc>
		<defs><marker id="ablation-rival-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<text class="viz-axis-label" x="14" y="27">DISTANT EVIDENCE: ACCURACY (%)</text>
		<text class="viz-label" x="14" y="47">Axis starts at 80%; bars show gains above 80.</text>
		<rect class="viz-plot-bg" x="60" y="74" width="256" height="180" rx="3"></rect>
		<path class="viz-gridline" d="M60 74V254 M124 74V254 M188 74V254 M252 74V254 M316 74V254"></path>
		<path class="viz-axis" d="M60 74V254H316"></path>
		<text class="viz-callout" x="14" y="104">A</text>
		<circle class="viz-operating-point" cx="60" cy="100" r="4"></circle>
		<text class="viz-callout" x="75" y="104">80</text>
		<text class="viz-callout" x="14" y="160">C</text>
		<rect class="viz-node" x="60" y="147" width="179.2" height="18" rx="2"></rect>
		<text class="viz-callout" x="247" y="160">82.8</text>
		<text class="viz-callout" x="14" y="216">F</text>
		<rect class="viz-node viz-node--focus" x="60" y="203" width="192" height="18" rx="2"></rect>
		<text class="viz-callout" x="260" y="216">83</text>
		<text class="viz-label" x="60" y="273" text-anchor="middle">80</text>
		<text class="viz-label" x="124" y="273" text-anchor="middle">81</text>
		<text class="viz-label" x="188" y="273" text-anchor="middle">82</text>
		<text class="viz-label" x="252" y="273" text-anchor="middle">83</text>
		<text class="viz-label" x="316" y="273" text-anchor="middle">84</text>
		<path class="viz-operating-guide" d="M60 284V292H252V284"></path>
		<text class="viz-callout" x="156" y="311" text-anchor="middle">F − A = 3 points</text>
		<rect class="viz-node" x="14" y="332" width="332" height="78" rx="4"></rect>
		<text class="viz-callout" x="26" y="352">Distant: F − C = 0.2 points</text>
		<text class="viz-callout" x="26" y="373">Short: F − C = 0.1 points</text>
		<text class="viz-label" x="26" y="395">Difference of gaps = 0.1 points</text>
		<rect class="viz-node viz-node--focus" x="14" y="430" width="332" height="64" rx="4"></rect>
		<text class="viz-axis-label" x="26" y="450">PREDECLARED MECHANISM PREDICTION</text>
		<text class="viz-callout" x="26" y="477">Distant gain ≥ 2; short gain ≤ 0.5 points</text>
		<text class="viz-label" x="180" y="521" text-anchor="middle">Hypothetical means; uncertainty is not supplied.</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> the full model beats A by three points but beats matched C by only 0.2. These comparisons support different claims. <span class="visual-reader-check" data-reader-check="ablation-rival-hypotheses"><strong>Check:</strong> Does a three-point baseline gain establish the mechanism when the matched-control gap is only 0.2 points?</span></figcaption>
</figure>

<p class="diagram-source">References: <a href="https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm">NIST blocking guidance</a> and <a href="https://arxiv.org/abs/1707.05589">Melis et al.</a></p>

## Specify the experiment before running it

| Arm | Input or intervention | Purpose |
| --- | --- | --- |
| A | Original smaller architecture | Reproduction anchor, not capacity-matched control |
| F | Full content input, $\lambda=1$ | Claimed mechanism |
| C | Position-only input, $\lambda=0$ | Usable matched replacement |
| Half | Half content, half position | Predeclared mechanism-strength test |

For routing arms, use the same gate on $\lambda c_{\rm norm}+(1-\lambda)p_{\rm norm}$.
Compute the same input-blend path and verify actual resource differences.
Predeclare the expected strength and regime pattern, not a pattern chosen after inspecting scores.

Three paired runs per arm, each capped at four GPU-hours including tuning, training and evaluation, cost at most 48 GPU-hours.
Both regimes evaluate the same checkpoints; they do not double the run count.
Count failed and stabilizing trials against the cap.

Record config identity, initialization policy, data/evaluation versions, hardware, tokens, FLOPs, wall time and tuning opportunities.
Choose the resource contract relevant to the claim and disclose residual differences.
Parameters, FLOPs, utilization and wall time cannot generally all match exactly.

Retain paired run/item outputs, primary and secondary measures, uncertainty method and stopping rule.
Three seeds are an affordable pilot plan, not a universal adequacy rule.

## Change the intervention

### 1. Removal causes divergence

**New task:** deleting the router disrupts optimization instead of yielding a lower converged score.

**Picture and change:** retain deletion as a diagnostic, not a clean mechanism estimate.
Inspect loss/gradient trajectories and compare stable matched replacements with equal predeclared stabilization opportunities.

**Check:** recovery by a stable control shows intervention sensitivity, not necessity of content routing.

**Use when:** removal changes initialization, residual scale or numerical stability.

### 2. Two components may interact

**New task:** routing and a normalization change might matter jointly.

**Picture and change:** use a reviewed $2\times2$ on/off design within a revised authorized run allocation.
Do not silently add another study.

**Check:** for scores $Y_{00}=80$, $Y_{10}=80.2$, $Y_{01}=80.1$, $Y_{11}=83$:

$$
Y_{11}-Y_{10}-Y_{01}+Y_{00}=2.7\text{ points}.
$$

**Use when:** one-at-a-time removals cannot separate main effects from interaction.

## Failure and transfer

Adding unused weights does not match usable capacity or optimization.
A correlated mediator score without a matched quality effect does not establish that mediator's causal role.

What if average performance improves, but the effect is absent where theory predicts the largest benefit?

<details>
<summary>Check your answer</summary>

Retain the performance result while narrowing the tested mechanism claim according to uncertainty and control validity.
Do not invent a replacement boundary prediction after seeing results.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Clarify the claim

Separate quality, efficiency, convergence, robustness and transfer claims.
Check data, parameters, compute, tuning and regime differences before interpreting a gain.

## A strong ablation sequence

1. Reproduce results across declared seeds and estimate relevant uncertainty.
2. Match the chosen resource contract.
3. Remove the component with other implementation choices fixed.
4. Add a usable simple replacement.
5. Vary mechanism strength under controlled resources.
6. Test predicted boundary regimes.
7. Measure predicted mediators without equating correlation with causation.
8. Replicate across tasks only as broadly as the claim requires.

<span id="what-an-l4-answer-sounds-like"></span>

## An incomplete removal

Removing a layer tests that intervention, but can also change capacity, optimization and numerical behavior.

<span id="what-an-l5-answer-adds"></span>

## A controlled contrast

State which observation would falsify the proposed explanation and which rival factors remain.

<span id="what-an-l6-answer-adds"></span>

## Scope and identification

Ask whether the effect survives scale and cheaper interventions.
Even a matched replacement can retain or introduce another mechanism.
No single ablation automatically identifies a unique universal cause.

<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Useful evidence

Distinguish final performance from mechanism evidence and account for multiple comparisons and post-selection.
An answer format does not guarantee a hiring outcome.

<span id="tells-that-get-you-down-leveled"></span>

## Incomplete reporting

One seed, unequal hidden tuning, or only the best configuration cannot establish a robust controlled comparison.
Feature importance is not automatically a causal ablation.

## Common follow-ups

- What if removing the component changes optimization stability?
- How do you match compute when architectures have different utilization?
- What mediator would support the proposed mechanism?
- How many seeds are enough?
- What result would make you reject the claim despite a positive average gain?

## Sources

- [Melis et al., controlled language-model comparisons](https://arxiv.org/abs/1707.05589).
- [We need to talk about random seeds](https://arxiv.org/abs/2210.13393).
- [NIST randomized block designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm).
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [bias and variance of estimators](/concepts/bias-variance-of-estimators/), and [critique an ML paper](/questions/critique-ml-paper/).*

</div>
