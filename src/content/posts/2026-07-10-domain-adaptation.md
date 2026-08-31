---
title: "Domain adaptation"
description: "Transfer a model across related but shifted data distributions without assuming unlabeled target data makes the problem identifiable."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Training on one distribution and deploying on a related but shifted one is the normal case, not the exception: a fraud model meets new fraud, a medical model meets a new hospital's scanner, a speech model meets a new accent. Domain adaptation transfers a predictor from a source distribution $P_s(X,Y)$ to a target $P_t(X,Y)$ where the label space is usually the same but the inputs, prevalences, or input-label relationship have moved. The first job is to name which of those moved, because that decides whether the problem is even solvable from the data you have.

## Shift types

- **Covariate shift:** $P(X)$ changes while $P(Y\mid X)$ is stable.
- **Label shift:** $P(Y)$ changes while $P(X\mid Y)$ is stable.
- **Concept shift:** $P(Y\mid X)$ changes; unlabeled target data alone is generally not enough.

Naming the assumed shift determines which correction is defensible.

**Learning objective:** distinguish covariate, label, and concept shift by the factor that changes, then decide whether unlabeled target inputs contain enough evidence to adapt.

<!-- visual:domain-adaptation-shift-assumptions -->
<figure class="learning-figure plot-panel" aria-labelledby="domain-adaptation-visual-title">
	<p class="visual-kicker">Assumption map</p>
	<p class="visual-title" id="domain-adaptation-visual-title">What changed, what stayed fixed, and what can target inputs reveal?</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 390" role="img" aria-labelledby="domain-adaptation-svg-title domain-adaptation-svg-desc">
			<title id="domain-adaptation-svg-title">Evidence available under three kinds of dataset shift</title>
			<desc id="domain-adaptation-svg-desc">Three rows compare source and target distributions. Under covariate shift, P of X changes while P of Y given X stays fixed, so unlabeled target X can support density-ratio weighting when source and target inputs overlap. Under label shift, P of Y changes while P of X given Y stays fixed, so unlabeled target X plus an identifiable source confusion matrix can estimate target class priors. Under concept shift, P of Y given X changes. The same unlabeled target X is compatible with multiple target labeling rules, so target labels or additional structural assumptions are required.</desc>
			<defs>
				<marker id="domain-adaptation-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
					<path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path>
				</marker>
			</defs>
			<rect class="viz-plot-bg" x="8" y="12" width="344" height="360" rx="4"></rect>
			<text class="viz-axis-label" x="20" y="34">ASSUMED SHIFT</text>
			<text class="viz-axis-label" x="132" y="34">DISTRIBUTION FACTORS</text>
			<text class="viz-axis-label" x="132" y="49">changed / fixed</text>
			<text class="viz-axis-label" x="263" y="34">EVIDENCE</text>
			<rect class="viz-node viz-node--input" x="18" y="64" width="96" height="82" rx="4"></rect>
			<text class="viz-node-label" x="66" y="88">COVARIATE</text>
			<text class="viz-node-value" x="66" y="108">P(X) changes</text>
			<text class="viz-label" x="66" y="127" text-anchor="middle">input mix moves</text>
			<text class="viz-callout" x="132" y="87">CHANGED  P(X)</text>
			<text class="viz-label" x="132" y="108">FIXED  P(Y | X)</text>
			<path d="M222 104H248" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8;marker-end:url(#domain-adaptation-arrow)"></path>
			<text class="viz-callout" x="260" y="84">target X helps</text>
			<text class="viz-label" x="260" y="104">density-ratio</text>
			<text class="viz-label" x="260" y="120">weights, if</text>
			<text class="viz-label" x="260" y="136">support overlaps</text>
			<path class="viz-gridline" d="M18 160H342"></path>
			<rect class="viz-node viz-node--focus" x="18" y="174" width="96" height="82" rx="4"></rect>
			<text class="viz-node-label" x="66" y="198">LABEL</text>
			<text class="viz-node-value" x="66" y="218">P(Y) changes</text>
			<text class="viz-label" x="66" y="237" text-anchor="middle">class mix moves</text>
			<text class="viz-callout" x="132" y="197">CHANGED  P(Y)</text>
			<text class="viz-label" x="132" y="218">FIXED  P(X | Y)</text>
			<path d="M222 214H248" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8;marker-end:url(#domain-adaptation-arrow)"></path>
			<text class="viz-callout" x="260" y="194">target X +</text>
			<text class="viz-label" x="260" y="214">identifiable</text>
			<text class="viz-label" x="260" y="230">source confusion</text>
			<text class="viz-label" x="260" y="246">matrix helps</text>
			<path class="viz-gridline" d="M18 270H342"></path>
			<rect x="18" y="284" width="96" height="72" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5;stroke-dasharray:5 3"></rect>
			<text class="viz-node-label" x="66" y="308">CONCEPT</text>
			<text class="viz-node-value" x="66" y="328">P(Y | X) changes</text>
			<text class="viz-label" x="66" y="347" text-anchor="middle">label rule moves</text>
			<text class="viz-callout" x="132" y="307">CHANGED  P(Y | X)</text>
			<text class="viz-label" x="132" y="328">target P(X) alone</text>
			<text class="viz-label" x="132" y="344">cannot reveal it</text>
			<path d="M222 324H248" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.8;stroke-dasharray:5 3;marker-end:url(#domain-adaptation-arrow)"></path>
			<text class="viz-callout" x="260" y="307">need labels</text>
			<text class="viz-label" x="260" y="328">or additional</text>
			<text class="viz-label" x="260" y="344">structure</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> read each row from left to right. A correction is justified only after naming both the factor that changed and the one assumed fixed. Target inputs can expose a changed input mix and, with identifiability conditions, a changed class mix; they cannot reveal a new label rule by themselves. Original synthesis checked against <a href="https://jmlr.org/papers/v8/sugiyama07a.html">Sugiyama et al. (2007)</a>, <a href="https://proceedings.mlr.press/v80/lipton18a.html">Lipton et al. (2018)</a>, and <a href="https://proceedings.mlr.press/v9/david10a.html">Ben-David et al. (2010)</a>.</figcaption>
</figure>

## Approaches

- Importance weighting under a covariate- or label-shift assumption
- Fine-tuning on a small labeled target set
- Feature alignment with a discrepancy or adversarial objective
- Self-training with confidence and calibration controls
- Domain-specific normalization or adapters
- Robust optimization across the environments you can observe

## Evaluation

Use a true target-domain holdout and report the slices that matter. Validate calibration, not just ranking or accuracy, and measure negative transfer: adaptation can help the aggregate while hurting a target subgroup or eroding source performance.

## In an interview

1. Define source, target, labels, and how much target supervision you have.
2. State the shift assumption.
3. Establish source-only and target-labeled baselines.
4. Pick the simplest method the evidence justifies.
5. Monitor drift and collect the labels that distinguish concept shift from covariate shift.

## Common confusions

- **"Align the feature distributions and the task transfers."** Alignment can mix classes or erase the predictive structure you needed.
- **"Unlabeled target data solves domain shift."** Not when the label relationship itself changed.
- **"Fine-tuning always helps."** A small, biased target set can cause negative transfer and calibration failure.

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [epistemic versus aleatoric uncertainty](/concepts/epistemic-vs-aleatoric-uncertainty/), and [calibration](/concepts/calibration/).*
