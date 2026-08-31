---
title: "Logistic regression"
description: "Linear regression for binary classification: pass a linear combination through a sigmoid, train by maximum likelihood. Still the strongest non-trivial baseline for tabular classification."
date: "2026-04-17"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Logistic regression models $p(y = 1 \mid x) = \sigma(w^\top x + b)$ where $\sigma(z) = 1 / (1 + e^{-z})$ is the sigmoid. Trained by maximum likelihood = minimizing binary cross-entropy.

Logistic regression is the **first model you should try** on any tabular classification problem. It is interpretable, calibrated by default (when trained on representative data), fast to fit, and competitive with much fancier methods on high-quality features. Most "production tabular models" at large companies have a strong logistic baseline they need to beat.

It is also the canonical example of a [generalized linear model](/concepts/exponential-family/) and the building block for softmax regression, neural network output layers, and many fairness / calibration analyses.

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

<!-- visual:logit-odds-probability-map -->
<figure class="learning-figure" aria-labelledby="logit-map-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="logit-map-title">See how a linear score becomes multiplicative odds and a saturating probability.</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 290" role="img" aria-labelledby="logit-map-svg-title logit-map-svg-desc">
			<title id="logit-map-svg-title">Linear logits mapped to odds and sigmoid probabilities</title>
			<desc id="logit-map-svg-desc">A sigmoid curve maps the linear score z from negative values to probabilities near zero and positive values to probabilities near one. Five equally spaced scores, minus two through two, are marked. Their probabilities are 0.12, 0.27, 0.50, 0.73, and 0.88, while their odds relative to the preceding unit scale by the constant factor e. A dashed vertical line at score zero marks even odds, probability one half, and the usual decision boundary.</desc>
			<rect class="viz-plot-bg" x="34" y="26" width="310" height="204" rx="4"></rect>
			<path class="viz-gridline" d="M34 128H344M34 77H344M34 179H344"></path>
			<path class="viz-axis" d="M34 26V230H344"></path>
			<text class="viz-label" x="16" y="231">0</text>
			<text class="viz-label" x="8" y="132">0.5</text>
			<text class="viz-label" x="16" y="31">1</text>
			<text class="viz-axis-label" transform="translate(16 174) rotate(-90)">probability p</text>
			<path d="M34 214.5C45 212.5 55 209.5 65 205.7S111 186 127 175.1S174 140 189 128S235 91.8 251 80.9S298 54.4 313 50.3S333 43.5 344 41.5" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round"></path>
			<path class="viz-operating-guide" d="M189 26V230"></path>
			<circle class="viz-operating-point" cx="65" cy="205.7" r="5"></circle>
			<circle class="viz-operating-point" cx="127" cy="175.1" r="5"></circle>
			<path d="M189 121L196 128L189 135L182 128Z" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
			<circle class="viz-operating-point" cx="251" cy="80.9" r="5"></circle>
			<circle class="viz-operating-point" cx="313" cy="50.3" r="5"></circle>
			<text class="viz-callout" x="70" y="197">p=.12</text>
			<text class="viz-callout" x="132" y="167">p=.27</text>
			<text class="viz-callout" x="197" y="122">p=.50</text>
			<text class="viz-callout" x="244" y="72" text-anchor="end">p=.73</text>
			<text class="viz-callout" x="306" y="42" text-anchor="end">p=.88</text>
			<text class="viz-axis-label" x="197" y="145">decision boundary</text>
			<path d="M65 238V244M127 238V244M189 238V244M251 238V244M313 238V244M65 241H313" style="fill:none;stroke:var(--c-text-soft);stroke-width:1.4"></path>
			<text class="viz-axis-label" x="65" y="258" text-anchor="middle">z=-2</text>
			<text class="viz-axis-label" x="127" y="258" text-anchor="middle">-1</text>
			<text class="viz-axis-label" x="189" y="258" text-anchor="middle">0</text>
			<text class="viz-axis-label" x="251" y="258" text-anchor="middle">1</text>
			<text class="viz-axis-label" x="313" y="258" text-anchor="middle">2</text>
			<text class="viz-label" x="65" y="276" text-anchor="middle">odds .14</text>
			<text class="viz-label" x="127" y="276" text-anchor="middle">.37</text>
			<text class="viz-label" x="189" y="276" text-anchor="middle">1</text>
			<text class="viz-label" x="251" y="276" text-anchor="middle">2.72</text>
			<text class="viz-label" x="313" y="276" text-anchor="middle">7.39</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> move one equal step along the linear score z = w<sup>T</sup>x + b. Each step multiplies the odds by the same factor e (0.14, 0.37, 1, 2.72, 7.39), but the sigmoid compresses the resulting probabilities near 0 and 1. At z = 0, odds are 1:1 and p = 0.5, so the usual threshold is the linear boundary w<sup>T</sup>x + b = 0. Original schematic checked against <a href="https://www.statlearning.com/"><cite>An Introduction to Statistical Learning</cite></a> and the <a href="https://cs229.stanford.edu/notes2022fall/main_notes.pdf?forcedefault=true">Stanford CS229 notes</a>.</figcaption>
</figure>

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
