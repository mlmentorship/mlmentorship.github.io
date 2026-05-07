---
title: "Decision trees"
description: "Recursively split the feature space along axis-aligned thresholds chosen to maximize a purity criterion. The base learner of GBDT and random forests."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A **decision tree** partitions the feature space by a sequence of axis-aligned threshold tests on individual features, assigning a constant prediction (class probability for classification, mean target for regression) to each leaf. Trained greedily by choosing splits that maximize information gain or equivalently minimize a purity / impurity criterion.

## Why it matters

Single trees are rarely the production model. Variance is too high. But they are the building block of the dominant tabular learners: **gradient boosting** (xgboost, lightgbm, catboost) and **random forests**. Understanding tree training is the prerequisite for using or tuning either.

Trees naturally handle missing values, mixed-type features, non-linear interactions, and require essentially no feature scaling. This is why they remain the strongest baseline on heterogeneous tabular data.

## The algorithm (CART, [Breiman et al., 1984](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418))

For each node:

1. For each feature $j$ and each candidate threshold $t$:
   - Split data into $\{x : x_j \le t\}$ (left child) and $\{x : x_j > t\}$ (right child).
   - Compute the impurity of each child (Gini, entropy, or MSE).
   - Compute the weighted impurity reduction.
2. Pick the $(j, t)$ with maximum reduction.
3. Recurse on each child until a stopping criterion (max depth, min samples per leaf, no further reduction).

The greedy choice is locally optimal but not globally; finding the globally optimal tree is NP-hard.

## Split criteria

For classification at a node with class proportions $p_k$:

- **Gini impurity**: $1 - \sum_k p_k^2$. Equals expected error of random labeling proportional to $p$.
- **Entropy**: $-\sum_k p_k \log p_k$. Information gain = parent entropy minus weighted child entropies.

Gini and entropy give nearly identical splits in practice; Gini is slightly cheaper.

For regression: variance of the target within the node, equivalent to MSE under constant prediction.

## Stopping and pruning

Stop splitting when:

- Max depth reached.
- Number of samples at the node falls below a threshold.
- Best impurity reduction is below a threshold.
- All samples have the same target.

**Pruning**: train deep, then collapse branches that don't reduce held-out error. Cost-complexity pruning (CCP) trades depth against penalty $\alpha \times |\text{leaves}|$.

In practice with ensembles (boosting, forests), individual trees are kept shallow (depth 6–8 for boosting, depth limited or full for forests with bagging variance averaging).

## Strengths and weaknesses

**Strengths:**

- Handles non-linear interactions for free.
- No feature scaling needed.
- Handles missing values (with surrogate splits or built-in handling in xgboost/lightgbm).
- Handles mixed numeric / categorical (with appropriate encoding).
- Interpretable: every prediction is a path of explicit rules.

**Weaknesses:**

- High variance: small data perturbations change splits dramatically.
- Greedy: locally optimal, not globally.
- Axis-aligned splits make it hard to capture rotated decision boundaries (need oblique trees or feature engineering).
- Single trees are weak. Usually combined into ensembles.

## Categorical features

Two main approaches:

- **One-hot**: each level becomes a binary feature. Slow; trees grow many useless branches.
- **Native handling** (lightgbm, catboost): pre-sort levels by target mean, then split as if the feature were numeric. Much faster and often more accurate.

## Common pitfalls

- **Letting trees grow unbounded on small data.** Memorizes training set; high variance.
- **Treating categorical features as numeric without one-hot or native handling.** Tree splits are ordered; "encoding" a categorical with arbitrary integer codes imposes a meaningless order.
- **Comparing single trees against ensembles unfairly.** Single trees should not beat boosted forests; check you are evaluating the right model class.
- **Reading feature importance from a single split.** Importance from a single tree is noisy; use ensemble averages or permutation importance.

## Related

- [Random forests](/reference/random-forests/). Bagged ensembles of trees.
- [Gradient boosting](/reference/gradient-boosting/). Boosted ensembles of trees.
