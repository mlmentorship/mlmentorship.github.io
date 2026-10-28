---
title: "Gradient boosting: fit the next correction"
description: "Trace a residual stump, recompute losses after shrinkage, and distinguish loss gradients, binary logits, and regularized Newton leaf weights."
date: "2026-02-21"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Gradient boosting adds a learner to an existing prediction function.
Its training targets are negative loss gradients at current predictions, not generally the original labels.
Shrinkage scales the new contribution before addition.

**Task:** ordered features are $x=[1,2,3,4]$, targets $y=[2,2,8,8]$, and current prediction $F_0=5$.
Use per-example loss $(y-F)^2/2$ and learning rate $\eta=0.5$.
Fit one residual stump, update all predictions, and recompute total loss.
This is a training fixture, not a held-out improvement claim.

## The algorithm [(Friedman, 2001)](https://www.jstor.org/stable/2699986)

Initialize $F_0=\arg\min_c\sum_iL(y_i,c)$.
For half-squared loss this is the mean; for Bernoulli loss it is finite log-odds when both classes occur.
Then repeat:

1. Compute the negative gradient (pseudo-residuals) at each training point:
   $$
   r_{i,t} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}.
   $$
2. Fit a regression tree $h_t$ to $\{(x_i, r_{i,t})\}$.
3. If required, optimize leaf values against the original objective; squared-error residual means already supply its reference correction.
4. Update: $F_t = F_{t-1} + \eta \cdot h_t$.

Our gradients $F_0-y=[3,3,-3,-3]$ give pseudo-residuals $[-3,-3,3,3]$.
A stump at $x=2.5$ fits leaf values $-3$ and $3$.
Adding $\eta h=[-1.5,-1.5,1.5,1.5]$ gives $F_1=[3.5,3.5,6.5,6.5]$.
SSE falls from $4(3^2)=36$ to $4(1.5^2)=9$; summed half-squared loss falls from $18$ to $4.5$.
This split is the diagram's “sample ≤ 2” rule for these ordered features.

**Learning objective:** trace one squared-error boosting round from the current prediction, through signed residual targets, to the learning-rate-scaled correction.

<!-- visual:gradient-boosting-residual-update -->
<figure class="learning-figure plot-panel" aria-labelledby="gradient-boosting-update-title">
	<p class="visual-kicker">One boosting round</p>
	<p class="visual-title" id="gradient-boosting-update-title">The next tree predicts how to move the current ensemble.</p>
	<svg viewBox="0 0 360 500" role="img" aria-labelledby="gradient-boosting-svg-title gradient-boosting-svg-desc">
		<title id="gradient-boosting-svg-title">A squared-error gradient-boosting residual update</title>
		<desc id="gradient-boosting-svg-desc">Four ordered samples have targets 2, 2, 8, and 8. The initial constant ensemble predicts 5 for every sample, producing residuals negative 3 for the first pair and positive 3 for the second pair. A decision stump splits after sample two and predicts those residual values. With learning rate one half, the stump contributes negative 1.5 on the left and positive 1.5 on the right. Adding that correction changes the predictions to 3.5, 3.5, 6.5, and 6.5, cutting every remaining target gap in half.</desc>
		<defs>
			<marker id="gradient-boosting-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-focus-stroke)"></path></marker>
		</defs>
		<text class="viz-axis-label" x="18" y="20">1 · CURRENT ENSEMBLE F₀</text>
		<rect class="viz-plot-bg" x="42" y="30" width="286" height="122" rx="4"></rect>
		<path class="viz-gridline" d="M42 50H328M42 91H328M42 132H328"></path>
		<path d="M53 91H317" style="fill:none;stroke:var(--viz-edge);stroke-width:2.5"></path>
		<text class="viz-callout" x="313" y="84" text-anchor="end">F₀ = 5</text>
		<g style="fill:var(--viz-neutral-bg);stroke:var(--viz-input-stroke);stroke-width:2.5">
			<circle cx="76" cy="132" r="6"></circle><circle cx="137" cy="132" r="6"></circle><circle cx="223" cy="50" r="6"></circle><circle cx="284" cy="50" r="6"></circle>
		</g>
		<g style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#gradient-boosting-arrow)">
			<path d="M76 94V122"></path><path d="M137 94V122"></path><path d="M223 88V60"></path><path d="M284 88V60"></path>
		</g>
		<text class="viz-callout" x="106" y="113" text-anchor="middle">r = −3 ↓</text>
		<text class="viz-callout" x="253" y="73" text-anchor="middle">r = +3 ↑</text>
		<text class="viz-label" x="185" y="148" text-anchor="middle">○ targets y = [2, 2, 8, 8]</text>
		<text class="viz-axis-label" x="18" y="182">2 · FIT A TREE TO r, NOT TO y</text>
		<path d="M180 218L99 250M180 218L261 250" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
		<rect class="viz-node viz-node--focus" x="120" y="194" width="120" height="34" rx="4"></rect>
		<text class="viz-callout" x="180" y="216" text-anchor="middle">sample ≤ 2?</text>
		<text class="viz-label" x="115" y="242" text-anchor="end">yes</text>
		<text class="viz-label" x="245" y="242">no</text>
		<rect class="viz-node viz-node--input" x="44" y="250" width="110" height="45" rx="4"></rect>
		<text class="viz-callout" x="99" y="269" text-anchor="middle">h₁ = −3</text>
		<text class="viz-label" x="99" y="286" text-anchor="middle">fit samples 1–2</text>
		<rect class="viz-node viz-node--input" x="206" y="250" width="110" height="45" rx="4"></rect>
		<text class="viz-callout" x="261" y="269" text-anchor="middle">h₁ = +3</text>
		<text class="viz-label" x="261" y="286" text-anchor="middle">fit samples 3–4</text>
		<text class="viz-callout" x="180" y="318" text-anchor="middle">η = 0.5 ⇒ add ηh₁ = −1.5 | +1.5</text>
		<text class="viz-axis-label" x="18" y="350">3 · ADD THE SHRUNKEN CORRECTION</text>
		<rect class="viz-plot-bg" x="42" y="360" width="286" height="122" rx="4"></rect>
		<path class="viz-gridline" d="M42 380H328M42 421H328M42 462H328"></path>
		<path d="M53 421H317" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:5 4"></path>
		<text class="viz-label" x="313" y="416" text-anchor="end">old F₀ = 5</text>
		<path d="M53 442H180V400H317" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3"></path>
		<text class="viz-callout" x="58" y="437">F₁ = 3.5</text>
		<text class="viz-callout" x="313" y="395" text-anchor="end">F₁ = 6.5</text>
		<g style="fill:var(--viz-neutral-bg);stroke:var(--viz-input-stroke);stroke-width:2.5">
			<circle cx="76" cy="462" r="6"></circle><circle cx="137" cy="462" r="6"></circle><circle cx="223" cy="380" r="6"></circle><circle cx="284" cy="380" r="6"></circle>
		</g>
		<g style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#gradient-boosting-arrow)">
			<path d="M76 445V453"></path><path d="M137 445V453"></path><path d="M223 397V389"></path><path d="M284 397V389"></path>
		</g>
		<text class="viz-label" x="180" y="497" text-anchor="middle">remaining residuals: −1.5 | +1.5; half as long</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> start at the flat prediction five. Under half-squared loss, target gaps are the next tree's negative-gradient labels. The stump predicts −3 and +3; shrinkage adds −1.5 and +1.5. Each remaining gap halves, while SSE falls from 36 to 9. Sources: <a href="https://www.jstor.org/stable/2699986">Friedman</a> and the <a href="https://xgboost.readthedocs.io/en/stable/tutorials/model.html">XGBoost tutorial</a>. <span class="visual-reader-check" data-reader-check="gradient-boosting-residual-update"><strong>Check:</strong> Why does the first tree predict -3 and +3 rather than the original targets 2 and 8?</span></figcaption>
</figure>

### Change the step size

**New task:** use $\eta=0.25$ with the same first stump.
Keep its fit and scale the corrections to $-0.75$ and $0.75$.
**Check:** predictions become $[4.25,4.25,5.75,5.75]$ and SSE becomes $4(2.25^2)=20.25$.
**Use when:** smaller updates are desired and extra rounds remain affordable.
Later rounds must recompute gradients at their changed ensemble predictions.

### Change the objective

**New task:** use labels $[0,0,1,1]$, Bernoulli cross-entropy, and initial raw logits $F=0$.
Probabilities are $p=\sigma(F)=0.5$, so negative gradients $y-p$ are $[-0.5,-0.5,0.5,0.5]$.
**Check:** they are not the regression gaps $[-3,-3,3,3]$ or the raw labels.
**Use when:** fitting probabilities instead of numeric responses; trees update logits before the sigmoid.

## Newton boosting (xgboost)

For logistic loss, $g_i=p_i-y_i$ and curvature $h_i=p_i(1-p_i)$.
In one fixed leaf, the quadratic approximation with L2 leaf penalty is

$$
G w+\frac12(H+\lambda)w^2,
\qquad G=\sum_i g_i,\quad H=\sum_i h_i.
$$

With positive denominator, differentiating gives $w=-G/(H+\lambda)$.
For the two negative labels above, $G=1$, $H=0.5$, and $\lambda=1$, hence $w=-2/3$.
This Newton weight differs from the first-order leaf mean $-0.5$.
Split structure scoring, leaf optimization, and learning-rate shrinkage are separate operations.
Curvature does not guarantee faster convergence or better generalization on every workload.

## Why it works

When $\sum_i r_i h_t(x_i)>0$, the learner points downhill locally in prediction space.
A finite step still needs suitable scale; approximate fitting is not a guaranteed useful correction on new data.
Each tree depends on the current ensemble, unlike independent averaging of separately fitted trees.

**Failure:** a flexible tree can fit label noise, lowering training loss while validation loss rises.
Choose early stopping on validation and restore the selected ensemble; selecting on test outcomes leaks evaluation information.

**Reconstruct:** what does the first reference update produce at $\eta=1$?

<details><summary>Answer</summary>

Predictions equal $[2,2,8,8]$, giving zero reference training SSE.
That exact fit says nothing about unseen points.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## The four implementations

| Library | Distinction to inspect in the chosen version |
|---------|----------------------|
| **XGBoost** | Regularized second-order objectives, split methods, and sparse/missing-value routing. |
| **LightGBM** | Histogram construction, leaf-wise growth, and declared categorical features. |
| **CatBoost** | Ordered boosting and ordered categorical statistics with their leakage controls. |
| sklearn `GradientBoostingClassifier` | Stagewise gradient boosting; compare separately with its histogram-based estimators. |

Benchmark actual data, hardware, objective, category policy, and quality constraints rather than ranking libraries universally.

## Hyperparameters that matter

| Control | What changes |
| --- | --- |
| Learning rate and rounds | Update magnitude and number of dependent corrections. |
| Depth or leaf count | Each learner's partition complexity. |
| Minimum leaf support or Hessian weight | Which candidate regions are admissible. |
| Row and feature sampling | Observations and candidate predictors used during fitting. |
| L1/L2 leaf penalties | Leaf optimization and associated split scoring. |

Names and defaults differ by library.
Learning rate and round count interact, but halving one and doubling the other is not an exact equivalence.

## When GBDT wins

It can be a useful candidate for heterogeneous features and threshold-based interactions.
Measure probability quality, task metrics, latency, and maintenance cost against meaningful alternatives.

## When GBDT loses

Raw unstructured inputs may need a different representation; tiny or shifted datasets can invalidate apparent gains.
Pairwise/listwise ranking needs its own objective and query grouping, not an ordinary regression interpretation.

## Common pitfalls

Fit preprocessing and target-based encodings without validation or test leakage.
Compare learning curves and restore best validated weights; neither fixed rounds nor early stopping automatically prevents overfitting.
See [decision trees](/concepts/decision-trees/) for weighted split mechanics and [logistic regression](/concepts/logistic-regression/) for Bernoulli probabilities.

</div>
