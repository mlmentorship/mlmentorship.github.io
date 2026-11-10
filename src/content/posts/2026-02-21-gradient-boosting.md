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
	<p class="visual-title" id="gradient-boosting-update-title">Fit current negative loss gradients, shrink the new learner, and recompute the objective</p><div class="visual-argument" data-visual-argument="gradient-boosting-residual-update" data-argument-kind="ledger" data-argument-composition="residual-step"><svg viewBox="0 0 360 2195" role="img" aria-label="Fit current negative loss gradients, shrink the new learner, and recompute the objective. Boosting fits negative gradients of the current loss, not the original labels again. Half-squared loss gives signed residuals; this stump fits minus three and three. Shrinkage halves those corrections, reducing training loss from eighteen to 4.5. Smaller steps change the update, not the first fitted stump. Logistic loss uses probability residuals and updates logits. Newton leaf weights also depend on curvature and regularization. Training improvement is not held-out generalization."><title>Fit current negative loss gradients, shrink the new learner, and recompute the objective</title><desc>Boosting fits negative gradients of the current loss, not the original labels again. Half-squared loss gives signed residuals; this stump fits minus three and three. Shrinkage halves those corrections, reducing training loss from eighteen to 4.5. Smaller steps change the update, not the first fitted stump. Logistic loss uses probability residuals and updates logits. Newton leaf weights also depend on curvature and regularization. Training improvement is not held-out generalization.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Fit a correction to the current loss</tspan><tspan x="20" dy="18">gradient</tspan></text><text x="20" y="105" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Toy: ordered features one to four;</tspan><tspan x="20" dy="18">targets two, two, eight, eight.</tspan><tspan x="20" dy="18">Current prediction five.</tspan></text><text x="20" y="222" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Sample 1: negative gradient -3</tspan></text><path d="M35 255L193.18181818181816 255" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M114.09090909090908 245L114.09090909090908 265" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="35" cy="255" r="6" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="20" y="337" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Sample 2: negative gradient -3</tspan></text><path d="M35 370L193.18181818181816 370" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M114.09090909090908 360L114.09090909090908 380" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="35" cy="370" r="6" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="20" y="452" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Sample 3: negative gradient 3</tspan></text><path d="M35 485L193.18181818181816 485" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M114.09090909090908 475L114.09090909090908 495" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="193.18181818181816" cy="485" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="20" y="567" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Sample 4: negative gradient 3</tspan></text><path d="M35 600L193.18181818181816 600" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M114.09090909090908 590L114.09090909090908 610" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="193.18181818181816" cy="600" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="20" y="780" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Residual stump: split after sample</tspan><tspan x="20" dy="18">two</tspan></text><rect x="45" y="865" width="110" height="58" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="100" y="900" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="100" dy="0">-3</tspan></text><text x="100" y="985" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="100" dy="0">Half-step</tspan><tspan x="100" dy="18">correction -1.5</tspan></text><rect x="205" y="865" width="110" height="58" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="260" y="900" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="260" dy="0">3</tspan></text><text x="260" y="985" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="260" dy="0">Half-step</tspan><tspan x="260" dy="18">correction 1.5</tspan></text><text x="20" y="1115" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Predictions move toward, not directly</tspan><tspan x="20" dy="18">onto, targets</tspan></text><text x="20" y="1195" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Samples one and two</tspan></text><path d="M114.09090909090908 1240L325 1240" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="166.8181818181818" cy="1240" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="245.9090909090909" cy="1240" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M245.9090909090909 1240L206.36363636363637 1240M214.36363636363637 1236L206.36363636363637 1240L214.36363636363637 1244" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-linejoin:round"></path><text x="20" y="1306" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Target 2; before five; after 3.5</tspan></text><text x="20" y="1345" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Samples three and four</tspan></text><path d="M114.09090909090908 1390L325 1390" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="325" cy="1390" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="245.9090909090909" cy="1390" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M245.9090909090909 1390L285.4545454545455 1390M277.4545454545455 1394L285.4545454545455 1390L277.4545454545455 1386" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-linejoin:round"></path><text x="20" y="1456" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Target 8; before five; after 6.5</tspan></text><text x="20" y="1550" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Summed half-squared training loss:</tspan><tspan x="20" dy="18">eighteen becomes 4.5</tspan></text><text x="20" y="1665" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Quarter steps: 4.25, 4.25, 5.75,</tspan><tspan x="20" dy="18">5.75; SSE 20.25. Later rounds</tspan><tspan x="20" dy="18">recompute gradients.</tspan></text><text x="20" y="1790" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Logistic loss at zero logits:</tspan><tspan x="20" dy="18">probability 0.5 gives negative</tspan><tspan x="20" dy="18">gradients minus/plus 0.5, not</tspan><tspan x="20" dy="18">minus/plus three.</tspan></text><text x="20" y="1930" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Logistic trees update logits. A</tspan><tspan x="20" dy="18">separate Newton leaf with gradient</tspan><tspan x="20" dy="18">sum one, curvature 0.5 and penalty</tspan><tspan x="20" dy="18">one has weight about minus 0.667, not</tspan><tspan x="20" dy="18">minus 0.5.</tspan></text><text x="20" y="2095" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Training improvement does not</tspan><tspan x="20" dy="18">establish held-out generalization.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> start at the flat prediction five. Under half-squared loss, target gaps are the next tree's negative-gradient labels. The stump predicts −3 and +3; shrinkage adds −1.5 and +1.5. Each remaining gap halves, while SSE falls from 36 to 9. Sources: <a href="https://www.jstor.org/stable/2699986">Friedman</a> and the <a href="https://xgboost.readthedocs.io/en/stable/tutorials/model.html">XGBoost tutorial</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="gradient-boosting-residual-update"><strong>Check:</strong> Why does the first tree predict -3 and +3 rather than the original targets 2 and 8?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="gradient-boosting-residual-update">Boosting fits negative gradients of the current loss, not the original labels again. Half-squared loss gives signed residuals; this stump fits minus three and three. Shrinkage halves those corrections, reducing training loss from eighteen to 4.5. Smaller steps change the update, not the first fitted stump. Logistic loss uses probability residuals and updates logits. Newton leaf weights also depend on curvature and regularization. Training improvement is not held-out generalization. <strong>Does logistic boosting reuse the regression gaps?</strong></span></figcaption>
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
