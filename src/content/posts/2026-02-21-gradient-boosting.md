---
title: "Gradient boosting (xgboost, lightgbm, catboost)"
description: "Train trees sequentially, each one fitting the gradient of the loss with respect to the current ensemble's prediction. The dominant tabular learner in 2026."
date: "2026-02-21"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Gradient boosting builds an ensemble $F(x) = \sum_t \eta \cdot h_t(x)$ one weak learner at a time. Each learner $h_t$, usually a small decision tree, fits the **negative gradient** of the current loss. The learning rate $\eta$ controls how much of that learner is added.

Gradient-boosted decision trees (GBDT) are the **dominant model class for tabular data in 2026**. xgboost, lightgbm, and catboost win the majority of Kaggle tabular competitions and are heavily used in production at scale (search ranking, ad CTR, fraud, credit risk). Knowing the algorithm at a level that distinguishes you from "I called `xgboost.fit`" is a core senior-ML expectation.

## The algorithm [(Friedman, 2001)](https://www.jstor.org/stable/2699986)

Initialize with a constant prediction $F_0$ (mean target for regression, log-odds prior for classification). Then for $t = 1, \dots, T$:

1. Compute the negative gradient (pseudo-residuals) at each training point:
   $$
   r_{i,t} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}.
   $$
2. Fit a regression tree $h_t$ to $\{(x_i, r_{i,t})\}$.
3. Optimize the leaf values to minimize $L$ in the new ensemble (line search per leaf).
4. Update: $F_t = F_{t-1} + \eta \cdot h_t$.

For squared error, $r_{i,t} = y_i - F_{t-1}(x_i)$. Literal residuals. For other losses (logistic, Huber, ranking) the residuals are the loss gradients.

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
		<g style="fill:var(--viz-surface);stroke:var(--viz-input-stroke);stroke-width:2.5">
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
		<g style="fill:var(--viz-surface);stroke:var(--viz-input-stroke);stroke-width:2.5">
			<circle cx="76" cy="462" r="6"></circle><circle cx="137" cy="462" r="6"></circle><circle cx="223" cy="380" r="6"></circle><circle cx="284" cy="380" r="6"></circle>
		</g>
		<g style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#gradient-boosting-arrow)">
			<path d="M76 445V453"></path><path d="M137 445V453"></path><path d="M223 397V389"></path><path d="M284 397V389"></path>
		</g>
		<text class="viz-label" x="180" y="497" text-anchor="middle">remaining residuals: −1.5 | +1.5; half as long</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> start at the flat prediction <var>F</var><sub>0</sub> = 5. Each signed arrow is both a target gap and, for squared error, a training label for the next tree. The stump fits −3 on the left and +3 on the right; shrinkage adds only half of that correction. The updated step <var>F</var><sub>1</sub> therefore moves toward every target without jumping all the way there, and each residual is halved. Original schematic checked against <a href="https://www.jstor.org/stable/2699986">Friedman's gradient-boosting formulation</a> and the <a href="https://xgboost.readthedocs.io/en/stable/tutorials/model.html">XGBoost boosted-tree tutorial</a>.</figcaption>
</figure>

## Newton boosting (xgboost)

xgboost uses second-order information: each new tree minimizes a Taylor expansion of the loss including both the gradient and Hessian (diagonal). This gives faster convergence and tighter leaf-value updates than first-order GBDT.

## Why it works

GBDT has a self-correcting property: each tree fixes the mistakes of the current ensemble. Combined with a small learning rate ($\eta = 0.05$ to $0.1$), this gives gradual, stable improvement.

The bias-variance picture flips compared to random forests:

- RF: low-bias trees, average → low variance.
- GBDT: shallow (high-bias) trees, but each adapts to current residuals → ensemble has low bias.

## The four implementations

| Library | Distinguishing feature |
|---------|----------------------|
| **xgboost** | Mature; great parallelization; sparsity-aware splits; default for many shops. |
| **lightgbm** | Histogram-based splits → much faster on large data; native categorical handling; leaf-wise growth. |
| **catboost** | Best out-of-the-box on categorical-heavy data; ordered boosting reduces target-leakage in categorical encodings. |
| sklearn `GradientBoostingClassifier` | Simple, slow on large data; mostly for teaching. |

For new projects in 2026: lightgbm for raw speed, catboost for categorical-heavy data, xgboost for everything else.

## Hyperparameters that matter

| Parameter | Typical | Effect |
|-----------|---------|--------|
| `learning_rate` | 0.05–0.1 | Smaller → more trees, better generalization, more compute. |
| `n_estimators` (or `num_round`) | 500–2000 | Use early stopping on validation. |
| `max_depth` | 4–8 | Most important regularizer. |
| `min_child_weight` / `min_data_in_leaf` | varies | Prevent overfitting to small leaves. |
| `subsample` | 0.7–0.9 | Stochastic gradient boosting; row sampling. |
| `colsample_bytree` | 0.5–1.0 | Feature subsampling per tree. |
| `reg_alpha`, `reg_lambda` | 0–10 | L1, L2 on leaf weights (xgboost). |

Use **early stopping** on a validation set: train until validation loss stops improving for $k$ rounds.

## When GBDT wins

- Mixed numeric + categorical features.
- Non-linear interactions matter.
- Modest sample size ($10^3$ to $10^7$).
- Heterogeneous feature scales.

## When GBDT loses

- High-dimensional unstructured data (text, images, audio): use neural nets.
- Truly tiny data ($n < 100$): logistic / linear with strong priors.
- Ranking with millions of items per query: dedicated learning-to-rank stacks (often still GBDT under the hood with pairwise / listwise losses).

## Common pitfalls

- **No early stopping → overfit.** Always stop on validation.
- **Tuning learning rate without retuning n_estimators.** They trade off; halving LR roughly doubles needed trees.
- **Default categorical handling = one-hot.** For lightgbm/catboost, declare categorical features explicitly to use native splits.
- **Comparing against a single tree.** That's not the right baseline; compare against RF and a strong logistic.
