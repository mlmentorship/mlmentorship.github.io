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

<!-- visual:ablation-rival-hypotheses -->
<figure class="learning-figure plot-panel" aria-labelledby="ablation-rival-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ablation-rival-title">Choose an ablation where the claimed mechanism and its strongest rival predict different results.</p>
	<svg viewBox="0 0 360 530" role="img" aria-labelledby="ablation-rival-svg-title ablation-rival-svg-desc">
		<title id="ablation-rival-svg-title">Ablation plan that distinguishes a mechanism from a resource confound</title>
		<desc id="ablation-rival-svg-desc">The full model improves by three percent, which is consistent with either the claimed mechanism or extra parameters, compute, and tuning. First test a matched control that keeps those resources but removes or replaces the mechanism. The mechanism hypothesis predicts that the gain disappears, while the resource hypothesis predicts that it remains. Then vary mechanism strength at fixed resources and test a boundary regime. The mechanism hypothesis predicts a dose or regime pattern specified in advance; the resource hypothesis predicts no such specific pattern. Prefer the explanation whose preregistered predictions match repeated-seed estimates and uncertainty.</desc>
		<defs><marker id="ablation-rival-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<rect class="viz-node viz-node--input" x="30" y="10" width="300" height="50" rx="4"></rect>
		<text class="viz-label" x="180" y="29" text-anchor="middle">OBSERVATION</text>
		<text class="viz-callout" x="180" y="49" text-anchor="middle">Full model beats baseline by 3%</text>
		<path d="M180 64V82H95V98" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#ablation-rival-arrow)"></path>
		<path d="M180 82H265V98" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#ablation-rival-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="10" y="102" width="165" height="70" rx="4"></rect>
		<text class="viz-axis-label" x="92.5" y="122" text-anchor="middle">H1: MECHANISM</text>
		<text class="viz-label" x="92.5" y="143" text-anchor="middle">component causes</text>
		<text class="viz-label" x="92.5" y="158" text-anchor="middle">the improvement</text>
		<rect class="viz-node" x="185" y="102" width="165" height="70" rx="4"></rect>
		<text class="viz-axis-label" x="267.5" y="122" text-anchor="middle">H2: RESOURCES</text>
		<text class="viz-label" x="267.5" y="143" text-anchor="middle">extra capacity, compute,</text>
		<text class="viz-label" x="267.5" y="158" text-anchor="middle">or tuning causes the gain</text>
		<text class="viz-axis-label" x="10" y="204">TEST 1 - MATCH THE RIVAL EXPLANATION</text>
		<rect class="viz-node" x="10" y="214" width="340" height="48" rx="4"></rect>
		<text class="viz-callout" x="180" y="234" text-anchor="middle">Same parameters, FLOPs, data, and tuning budget;</text>
		<text class="viz-label" x="180" y="251" text-anchor="middle">remove or replace the claimed mechanism</text>
		<path d="M92.5 266V282" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#ablation-rival-arrow)"></path>
		<path d="M267.5 266V282" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#ablation-rival-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="10" y="286" width="165" height="52" rx="4"></rect>
		<text class="viz-axis-label" x="92.5" y="306" text-anchor="middle">H1 PREDICTS</text>
		<text class="viz-callout" x="92.5" y="326" text-anchor="middle">gain disappears</text>
		<rect class="viz-node" x="185" y="286" width="165" height="52" rx="4"></rect>
		<text class="viz-axis-label" x="267.5" y="306" text-anchor="middle">H2 PREDICTS</text>
		<text class="viz-callout" x="267.5" y="326" text-anchor="middle">gain remains</text>
		<text class="viz-axis-label" x="10" y="370">TEST 2 - SEEK A MECHANISM-SPECIFIC PATTERN</text>
		<rect class="viz-node" x="10" y="380" width="340" height="48" rx="4"></rect>
		<text class="viz-callout" x="180" y="400" text-anchor="middle">At fixed resources, vary mechanism strength</text>
		<text class="viz-label" x="180" y="417" text-anchor="middle">and test a predicted boundary regime</text>
		<rect class="viz-node viz-node--focus" x="10" y="440" width="165" height="52" rx="4"></rect>
		<text class="viz-axis-label" x="92.5" y="460" text-anchor="middle">H1 PREDICTS</text>
		<text class="viz-callout" x="92.5" y="480" text-anchor="middle">specified dose pattern</text>
		<rect class="viz-node" x="185" y="440" width="165" height="52" rx="4"></rect>
		<text class="viz-axis-label" x="267.5" y="460" text-anchor="middle">H2 PREDICTS</text>
		<text class="viz-callout" x="267.5" y="480" text-anchor="middle">no specific pattern</text>
		<path class="viz-operating-guide" d="M10 498H350"></path>
		<text class="viz-axis-label" x="180" y="521" text-anchor="middle">DECIDE WITH REPEATED-SEED ESTIMATES + UNCERTAINTY</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> the original 3% result cannot choose between two explanations because both predict a gain. Spend the next runs on contrasts that split their predictions: a resource-matched replacement and a pre-specified dose or boundary test. Only then do repeated-seed estimates support the mechanism rather than merely restating the improvement. Original synthesis informed by the <a href="https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm">NIST treatment of nuisance factors</a> and <a href="https://arxiv.org/abs/1707.05589">Melis et al.'s controlled model comparisons</a>.</figcaption>
</figure>

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
