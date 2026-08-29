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
