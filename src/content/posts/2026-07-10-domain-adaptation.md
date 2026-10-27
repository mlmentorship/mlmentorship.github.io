---
title: "Domain adaptation"
description: "Transfer a model across related but shifted data distributions without assuming unlabeled target data makes the problem identifiable."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Domain adaptation transfers a predictor between source and target populations under explicit assumptions about what changed.
Specify the shared label space, sampling unit, available target supervision, time period, and deployment slices.
Unlabeled target inputs can reveal an input mixture without identifying a changed labeling rule.

Our teaching fixture has two input bins, fixed predictor errors $(0.1,0.4)$, and different population mixtures.
Compute target risk before choosing a correction.

## Shift types

- **Covariate shift:** $P(X)$ changes while $P(Y\mid X)$ is stable.
- **Label shift:** $P(Y)$ changes while $P(X\mid Y)$ is stable.
- **Concept shift:** $P(Y\mid X)$ changes; unlabeled target data alone is generally not enough.

Multiple factors can move together; a drift test on inputs does not establish an unchanged conditional distribution.

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
	<figcaption><strong>Read it this way:</strong> name both the changed factor and the invariant in each row. Input ratios require overlap and stable conditional labels. Label-prior estimation requires an identifiable, stable class-conditional statistic. Target inputs alone cannot reveal an unrestricted new labeling rule. Original synthesis checked against <a href="https://jmlr.org/papers/v8/sugiyama07a.html">Sugiyama et al. (2007)</a>, <a href="https://proceedings.mlr.press/v80/lipton18a.html">Lipton et al. (2018)</a>, and <a href="https://proceedings.mlr.press/v9/david10a.html">Ben-David et al. (2010)</a>.</figcaption>
</figure>

### Work the target risk

| Input bin | Source mass | Target mass | Conditional error | Target/source weight |
| --- | --- | --- | --- | --- |
| 0 | 0.8 | 0.2 | 0.1 | 0.25 |
| 1 | 0.2 | 0.8 | 0.4 | 4 |

Source risk is $0.8(0.1)+0.2(0.4)=0.16$.
Target risk is $0.2(0.1)+0.8(0.4)=0.34$.
Under stable $P(Y\mid X)$ and target support contained in source support,

$$
R_t(f)=\mathbb E_s\left[\frac{p_t(X)}{p_s(X)}\ell(f(X),Y)\right].
$$

The weighted source calculation gives $0.8(0.25)(0.1)+0.2(4)(0.4)=0.34$.
This verifies the algebra, not the real-world invariance assumption.
For 80/20 source rows, the weight-based effective sample size is

$$\frac{(80(0.25)+20(4))^2}{80(0.25)^2+20(4)^2}=\frac{400}{13}\approx30.77.$$

Large weights can reduce precision; clipping them changes the estimate and can introduce bias.

### Receive target labels

**New task:** use a small representative labeled target sample to test the assumed relationship.

**Picture and change:** reserve separate target evaluation; use adaptation labels for calibration or fitting, not both fitting and final scoring.

**Check:** target conditional error rates $(0.1,0.7)$ would instead give risk $0.2(0.1)+0.8(0.7)=0.58$.
Input reweighting alone still returns $0.34$, so it cannot repair that changed error relationship.
Allow sampling uncertainty before concluding that a finite sample establishes concept drift.

**Use when:** labeled evidence can discriminate assumptions that target inputs alone cannot test.

### Lose support or the labeling invariant

**New task:** deploy where the source has no observations or the target label rule differs.

**Picture and change:** restrict claims, collect justified labels, or abstain rather than inventing density ratios.

**Check:** if source mass on bin 1 is zero, its target mass $0.8$ has no source-label support.
With error $0.1$ in bin 0 and bounded loss, target risk can range from $0.02$ to $0.82$.
Separately, uniform inputs can accompany either $Y=X$ or $Y=1-X$.
Predictor $f(X)=X$ then has risk zero or one despite identical input distributions.

**Use when:** overlap or conditional invariance fails; matching feature distributions is not proof of transfer.

<!-- learning:core-end -->

<div data-learning-extension>

## Approaches

- Importance weighting requires the named invariant and support.
- Target fine-tuning requires representative supervision and separate evaluation.
- Feature alignment requires preserving label-relevant distinctions.
- Self-training can reinforce confident mistakes; validate its assumptions and calibration.
- Domain-specific normalization or adapters change the fitted model, not identifiability by themselves.
- Robust objectives compare the environments actually observed, not every possible future shift.

Under label shift, a fixed classifier statistic can satisfy $q_t=C\pi_t$.
Columns of $C$ are its stable prediction distributions given each class; $\pi_t$ contains target class priors.
For binary true-positive/false-positive rates $0.8/0.2$ and target positive-prediction rate $0.5$, the target prior is

$$\pi_t(1)=\frac{0.5-0.2}{0.8-0.2}=0.5.$$

If the two conditional rates coincide, this statistic cannot identify that prior.

## Evaluation

Compare source-only and target-supervised baselines on a true target holdout.
Report target slices, calibration, source retention, uncertainty, and negative transfer.
Reusing target evaluation labels for adaptation requires a new evaluation split.

## In an interview

Can unlabeled target inputs distinguish the two opposite label rules above?

<details><summary>Answer</summary>

No. Their input likelihoods agree exactly.
Target labels or additional structural assumptions are necessary.

</details>

## Common confusions

Aligned features or lower adaptation loss do not establish preserved class structure.
Biased target labels can worsen transfer even when their training loss decreases.

</div>

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [epistemic versus aleatoric uncertainty](/concepts/epistemic-vs-aleatoric-uncertainty/), and [calibration](/concepts/calibration/).*
