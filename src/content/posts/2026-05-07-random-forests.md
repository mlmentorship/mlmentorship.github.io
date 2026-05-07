---
title: "Random forests"
description: "Bag deep decision trees plus random feature subsets per split. Variance averaging beats any single tree; the dominant out-of-the-box ensemble before GBDT."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A **random forest** [(Breiman, 2001)](https://link.springer.com/article/10.1023/A:1010933404324) trains an ensemble of decision trees on bootstrap samples of the data ("bagging") with each split restricted to a random subset of features, then averages predictions (regression) or takes a majority vote (classification).

## Why it matters

Single decision trees are high-variance: small data shifts cause large changes in splits. Random forests average over many trees, dropping variance roughly as $1/\text{ensemble size}$. They are:

- A reliable **out-of-the-box baseline** for tabular data. Minimal tuning, strong performance.
- **Hard to overfit** with enough trees and reasonable depth limits.
- **Parallel-friendly**: trees train independently.

GBDT (xgboost, lightgbm, catboost) usually beats RF in modern tabular benchmarks, but RF remains useful as a fast baseline and for problems where you want low variance with minimal tuning.

## The two randomizations

1. **Bagging** (Bootstrap Aggregating): each tree trains on a bootstrap sample (sample with replacement, same size as original). Each tree sees ~63% of the unique training rows.
2. **Random feature subsets per split**: at each split, only $m$ randomly chosen features are considered as candidates. Standard $m = \sqrt{p}$ for classification, $m = p/3$ for regression.

Both reduce correlation between trees. Bagging alone gives "tree bagging"; the second randomization is what makes it a *random forest*.

## Out-of-bag (OOB) estimation

Each tree's bootstrap sample omits ~37% of the data. Predict each row using only trees that didn't see it (its **out-of-bag** trees) and average. This gives a free held-out estimate of generalization error. No separate validation set needed.

OOB estimates are typically very close to k-fold CV estimates. Useful for hyperparameter search without a separate split.

## Variance reduction analysis

For an ensemble of $T$ trees with pairwise correlation $\rho$ and per-tree variance $\sigma^2$:

$$
\mathrm{Var}(\bar f) = \rho \sigma^2 + \frac{1 - \rho}{T} \sigma^2.
$$

As $T \to \infty$, variance approaches $\rho \sigma^2$. The two randomizations work by *reducing $\rho$*; without them the trees are too similar and the second term dominates.

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

- [Decision trees](/reference/decision-trees/). The base learner.
- [Gradient boosting](/reference/gradient-boosting/). Sequential boosting alternative.
