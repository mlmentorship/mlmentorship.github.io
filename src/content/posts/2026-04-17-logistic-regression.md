---
title: "Logistic regression"
description: "Linear regression for binary classification: pass a linear combination through a sigmoid, train by maximum likelihood. Still the strongest non-trivial baseline for tabular classification."
date: "2026-04-17"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Logistic regression models $p(y = 1 \mid x) = \sigma(w^\top x + b)$ where $\sigma(z) = 1 / (1 + e^{-z})$ is the sigmoid. Trained by maximum likelihood = minimizing binary cross-entropy.

## Why it matters

Logistic regression is the **first model you should try** on any tabular classification problem. It is interpretable, calibrated by default (when trained on representative data), fast to fit, and competitive with much fancier methods on high-quality features. Most "production tabular models" at large companies have a strong logistic baseline they need to beat.

It is also the canonical example of a [generalized linear model](/reference/exponential-family/) and the building block for softmax regression, neural network output layers, and many fairness / calibration analyses.

## The model

For binary $y \in \{0, 1\}$:

$$
p(y = 1 \mid x; w, b) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}.
$$

The log-odds (logit) is linear in $x$:

$$
\log \frac{p(y=1 \mid x)}{p(y=0 \mid x)} = w^\top x + b.
$$

This is what "linear in the features" means here. Linear in the log-odds, not in the probability.

## Training

Negative log-likelihood (binary cross-entropy):

$$
L(w, b) = -\sum_{i=1}^{n} \big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \big].
$$

This loss is **convex** in $(w, b)$, so any local minimizer is global. No closed form (unlike linear regression); standard solvers:

- **L-BFGS** (default in scikit-learn): full-batch quasi-Newton.
- **SGD / Adam**: for very large datasets.
- **Newton-Raphson / IRLS**: classic statistical solver, fast for small problems.

Add L2 regularization (ridge) by appending $\lambda \|w\|^2$ to the loss; this is the default for most implementations.

## Multinomial / softmax regression

Generalize to $K$ classes: $p(y = k \mid x) \propto \exp(w_k^\top x + b_k)$. Loss is categorical cross-entropy. Output layer of every classification network is exactly this.

## Properties

- **Calibration**: when the linear log-odds assumption holds, predicted probabilities match empirical frequencies (well-calibrated by construction).
- **Interpretability**: $w_j$ is the change in log-odds per unit change in $x_j$ (holding others constant). $e^{w_j}$ is the odds ratio.
- **Decision boundary**: linear in feature space ($w^\top x + b = 0$). For non-linear boundaries, transform features first (interactions, polynomials, kernels). Equivalent to fitting in a transformed space.

## When to use vs. alternatives

| Setting | Logistic regression vs. alternative |
|---------|------------------------------------|
| Small-medium tabular, high-quality features | Logistic competitive with GBDT and neural nets |
| Sparse high-dimensional (text bag-of-words) | Logistic with L1 is excellent |
| Non-linear interactions matter | GBDT (xgboost, lightgbm) usually wins |
| Calibration matters, simple model required | Logistic is the answer |
| Large numbers of categorical features | Field-aware factorization machines or GBDT |
| Production scoring with tight latency | Logistic is the cheapest option |

## Common pitfalls

- **Forgetting to scale features.** Solvers converge faster and regularization is more meaningful when features are standardized.
- **Including the intercept in regularization.** Most implementations exclude it by default; if not, your model is biased toward predicting the prior near the boundary.
- **Comparing logistic against tree models on the same features.** Trees handle non-linear interactions automatically; logistic does not. Make features comparable (one-hot, target encoding) before claiming "X beats Y."
- **Using probability threshold 0.5 by default.** Pick the threshold from the precision-recall tradeoff at the deployment operating point.
