---
title: "Random forests"
description: "Bag deep decision trees plus random feature subsets per split. Variance averaging beats any single tree; the dominant out-of-the-box ensemble before GBDT."
date: "2026-01-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **random forest** [(Breiman, 2001)](https://link.springer.com/article/10.1023/A:1010933404324) trains an ensemble of decision trees on bootstrap samples of the data ("bagging") with each split restricted to a random subset of features, then averages predictions (regression) or takes a majority vote (classification).

Single decision trees are high-variance: small data shifts cause large changes in splits. Random forests average over many trees, shrinking the uncorrelated part of their variance roughly as $1/\text{ensemble size}$. They are:

- A reliable **out-of-the-box baseline** for tabular data. Minimal tuning, strong performance.
- **Hard to overfit** with enough trees and reasonable depth limits.
- **Parallel-friendly**: trees train independently.

GBDT (xgboost, lightgbm, catboost) usually beats RF in modern tabular benchmarks, but RF remains useful as a fast baseline and for problems where you want low variance with minimal tuning.

## The two randomizations

1. **Bagging** (Bootstrap Aggregating): each tree trains on a bootstrap sample (sample with replacement, same size as original). Each tree sees ~63% of the unique training rows.
2. **Random feature subsets per split**: at each split, only $m$ randomly chosen features are considered as candidates. Classic heuristics are $m = \sqrt{p}$ for classification and $m = p/3$ for regression.

Both reduce correlation between trees. Bagging alone gives "tree bagging"; the second randomization is what makes it a *random forest*.

## Out-of-bag (OOB) estimation

Each tree's bootstrap sample omits ~37% of the data. Predict each row using only trees that didn't see it (its **out-of-bag** trees) and average. This gives a free held-out estimate of generalization error. No separate validation set needed.

OOB estimates are typically very close to k-fold CV estimates. Useful for hyperparameter search without a separate split.

## Variance reduction analysis

For an ensemble of $T$ trees with pairwise correlation $\rho$ and per-tree variance $\sigma^2$:

$$
\mathrm{Var}(\bar f) = \rho \sigma^2 + \frac{1 - \rho}{T} \sigma^2.
$$

**Learning objective:** separate ensemble variance into the correlated floor that remains as trees are added and the uncorrelated share that averaging can remove.

<!-- visual:random-forest-correlation-floor -->
<figure class="learning-figure" aria-labelledby="random-forest-variance-title">
	<p class="visual-kicker">Why decorrelation matters</p>
	<p class="visual-title" id="random-forest-variance-title">More trees shrink only the uncorrelated share of variance.</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 330" role="img" aria-labelledby="random-forest-variance-svg-title random-forest-variance-svg-desc">
			<title id="random-forest-variance-svg-title">Random forest variance at one, four, and infinitely many trees</title>
			<desc id="random-forest-variance-svg-desc">A worked example sets pairwise tree correlation rho to one quarter and each tree's variance to one. Three stacked bars show total ensemble variance. At one tree, correlated variance is one quarter and uncorrelated variance is three quarters, totaling one. At four trees, the correlated floor remains one quarter while the uncorrelated share shrinks to three sixteenths, totaling seven sixteenths or about point four four. At infinitely many trees, the uncorrelated share vanishes but the one-quarter correlation floor remains. A note says bootstrap rows and random feature candidates lower rho and therefore lower the floor.</desc>
			<defs>
				<pattern id="random-forest-shared-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
					<rect width="7" height="7" style="fill:var(--viz-state-bg)"></rect>
					<path d="M0 0V7" style="stroke:var(--viz-state-stroke);stroke-width:2"></path>
				</pattern>
			</defs>
			<text class="viz-axis-label" x="15" y="24">VARIANCE, NORMALIZED BY &#963;&#178;</text>
			<path class="viz-gridline" d="M42 50H330M42 92.5H330M42 135H330M42 177.5H330"></path>
			<path class="viz-axis" d="M42 42V220H334"></path>
			<text class="viz-label" x="34" y="54" text-anchor="end">1.00</text>
			<text class="viz-label" x="34" y="181" text-anchor="end">0.25</text>
			<rect x="66" y="50" width="54" height="127.5" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="66" y="177.5" width="54" height="42.5" rx="3" style="fill:url(#random-forest-shared-hatch);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
			<rect x="157" y="145.6" width="54" height="31.9" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="157" y="177.5" width="54" height="42.5" rx="3" style="fill:url(#random-forest-shared-hatch);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
			<rect x="248" y="177.5" width="54" height="42.5" rx="3" style="fill:url(#random-forest-shared-hatch);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
			<text class="viz-callout" x="93" y="40" text-anchor="middle">1.00</text>
			<text class="viz-callout" x="184" y="136" text-anchor="middle">0.44</text>
			<text class="viz-callout" x="275" y="168" text-anchor="middle">0.25</text>
			<text class="viz-axis-label" x="93" y="242" text-anchor="middle">T = 1</text>
			<text class="viz-axis-label" x="184" y="242" text-anchor="middle">T = 4</text>
			<text class="viz-axis-label" x="275" y="242" text-anchor="middle">T &#8594; &#8734;</text>
			<path d="M118 108H306" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
			<text class="viz-label" x="310" y="104" text-anchor="end">uncorrelated share</text>
			<text class="viz-label" x="310" y="119" text-anchor="end">(1 &#8722; &#961;) / T shrinks</text>
			<path d="M118 201H306" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:2;stroke-dasharray:5 4"></path>
			<text class="viz-label" x="310" y="198" text-anchor="end" style="paint-order:stroke;stroke:var(--viz-canvas);stroke-width:4;stroke-linejoin:round">fixed floor &#961; = 0.25</text>
			<rect class="viz-node" x="28" y="264" width="304" height="48" rx="4"></rect>
			<text class="viz-callout" x="180" y="284" text-anchor="middle">bootstrap rows + random feature candidates</text>
			<text class="viz-label" x="180" y="302" text-anchor="middle">&#8594; less-aligned tree errors &#8594; lower &#961; and a lower floor</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> with &#961; = 0.25 and &#963;&#178; = 1, one tree has variance 0.25 + 0.75 = 1. Four trees keep the hatched 0.25 shared-error floor but shrink the remaining variance to 0.75 / 4, for 0.4375 total. More trees can erase the uncorrelated share, never the floor; bootstrap sampling and random feature candidates matter because they lower &#961; itself. Original schematic checked against <a href="https://doi.org/10.1023/A:1010933404324">Breiman (2001)</a> and <a href="https://hastie.su.domains/ElemStatLearn/">The Elements of Statistical Learning</a>.</figcaption>
</figure>

As $T \to \infty$, variance approaches $\rho \sigma^2$. The two randomizations work by *reducing $\rho$*; without them the trees are too similar and the correlated first term dominates.

## Bias-variance picture

- Each tree is grown deep (low bias, high variance).
- Averaging cuts variance.
- Bias of the ensemble = bias of a single tree (averaging unbiased-ish quantities).
- Bigger forests almost never overfit; you might get diminishing returns past 200–500 trees but rarely get worse.

## Hyperparameters that matter

| Parameter | Default | Notes |
|-----------|---------|-------|
| n_estimators | 100–500 | More is usually better; saturates around 200–500. |
| max_features | $\sqrt{p}$ (clf), $p/3$ (reg) | Smaller = more decorrelation, more bias. |
| max_depth / min_samples_leaf | None / 1 | Often left unbounded; constrain on huge data. |
| bootstrap | True | False = random subspace method, lighter on memory. |

## When to use vs. alternatives

- **Tabular baseline**: RF is the fastest "good enough" model.
- **GBDT** (xgboost, lightgbm): usually 1–3% better but needs more tuning.
- **Logistic regression**: better when the truth is approximately linear or interpretability matters.
- **Neural nets**: rarely beat tree-based models on small/medium tabular data.

## Common pitfalls

- **Comparing RF against a single tree.** Trivial; that's not the comparison.
- **Treating feature importances as causal.** They are correlational; correlated features split the importance among themselves.
- **Tuning on training accuracy.** Use OOB or CV; trees can fit training perfectly.
- **Skipping permutation importance.** Built-in importance favors high-cardinality features; permutation importance is more honest.

## Related

- [Decision trees](/concepts/decision-trees/). The base learner.
- [Gradient boosting](/concepts/gradient-boosting/). Sequential boosting alternative.
