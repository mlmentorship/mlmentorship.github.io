---
title: "Maximum likelihood estimation"
description: "The dominant statistical principle: pick parameters that make the observed data most probable. Reduces to minimizing cross-entropy for classification and MSE for Gaussian regression."
date: "2025-09-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

For a parametric family $p(x \mid \theta)$ and observed data $\{x_1, \dots, x_n\}$, the **maximum likelihood estimate (MLE)** is

$$
\hat\theta_\text{MLE} = \arg\max_\theta \prod_{i=1}^{n} p(x_i \mid \theta) = \arg\max_\theta \sum_{i=1}^{n} \log p(x_i \mid \theta).
$$

MLE underlies almost every modern ML loss function:

- Cross-entropy for classification = MLE under a categorical model.
- Mean-squared error = MLE under a Gaussian noise model.
- Negative log-likelihood for language models = MLE.

When you read "minimize the negative log-likelihood," you're reading MLE.

<!-- visual:mle-bernoulli-parameter-sweep -->
<figure class="learning-figure plot-panel" aria-labelledby="mle-sweep-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="mle-sweep-title">Read likelihood as a score over candidate parameters, not as a probability distribution over them</p>
	<svg viewBox="0 0 360 330" role="img" aria-labelledby="mle-sweep-svg-title mle-sweep-svg-desc">
		<title id="mle-sweep-svg-title">Bernoulli likelihood across candidate success probabilities</title>
		<desc id="mle-sweep-svg-desc">An original diagram starts with ten fixed Bernoulli observations: seven ones and three zeros. For each candidate success probability p, the likelihood is p to the seventh power times one minus p to the third power. A solid relative-likelihood curve rises from zero, reaches its unique maximum at p equals seven tenths, the observed fraction of successes, then returns to zero. A vertical guide and labelled point identify the maximum likelihood estimate. The curve is a score as p varies and is not a probability distribution over p.</desc>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="322" rx="6"></rect>
		<text class="viz-axis-label" x="16" y="25">fixed observations D</text>
		<g aria-label="Seven successes followed by three failures">
			<rect class="viz-node viz-node--focus" x="16" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="28.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="48" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="60.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="80" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="92.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="112" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="124.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="144" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="156.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="176" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="188.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="208" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="220.5" y="53">1</text>
			<rect class="viz-node" x="240" y="36" width="25" height="25" rx="3" style="stroke-dasharray:4 3"></rect><text class="viz-node-label" x="252.5" y="53">0</text>
			<rect class="viz-node" x="272" y="36" width="25" height="25" rx="3" style="stroke-dasharray:4 3"></rect><text class="viz-node-label" x="284.5" y="53">0</text>
			<rect class="viz-node" x="304" y="36" width="25" height="25" rx="3" style="stroke-dasharray:4 3"></rect><text class="viz-node-label" x="316.5" y="53">0</text>
		</g>
		<text class="viz-callout" x="16" y="82">candidate p → L(p; D) = p⁷(1 − p)³</text>
		<path class="viz-gridline" d="M45 170H340M45 222.5H340M45 275H340M45 170V275M192.5 170V275M340 170V275"></path>
		<path class="viz-axis" d="M45 170V275H340"></path>
		<path class="viz-roc-curve" d="M45 275 L59.8 275 L74.5 275 L89.3 275 L104 274.7 L118.8 273.8 L133.5 271.5 L148.3 266.7 L163 258.3 L177.8 245.6 L192.5 228.9 L207.2 209.5 L222 190.4 L236.8 175.7 L251.5 170 L266.3 176.5 L281 195.8 L295.8 223.9 L310.5 252.4 L325.3 270.9 L340 275"></path>
		<path class="viz-operating-guide" d="M251.5 170V275" style="stroke-dasharray:5 4"></path>
		<circle class="viz-operating-point" cx="251.5" cy="170" r="5"></circle>
		<text class="viz-callout" x="251.5" y="112" text-anchor="middle">maximum at p̂ = 7 / 10 = 0.7</text>
		<text class="viz-label" x="37" y="174" text-anchor="end">1</text>
		<text class="viz-label" x="37" y="226.5" text-anchor="end">0.5</text>
		<text class="viz-label" x="37" y="279" text-anchor="end">0</text>
		<text class="viz-label" x="45" y="294" text-anchor="middle">0</text>
		<text class="viz-label" x="192.5" y="294" text-anchor="middle">0.5</text>
		<text class="viz-label" x="251.5" y="294" text-anchor="middle">0.7</text>
		<text class="viz-label" x="340" y="294" text-anchor="middle">1</text>
		<text class="viz-axis-label" x="45" y="154">relative likelihood</text>
		<text class="viz-axis-label" x="192.5" y="316" text-anchor="middle">candidate success probability p</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> keep the ten observations fixed and move along the horizontal axis to try different values of <code>p</code>. Seven ones contribute <code>p⁷</code>; three zeros contribute <code>(1 − p)³</code>. Their product is largest at <code>p = 0.7</code>, so that candidate is the MLE. The curve compares parameter values; it is not a probability distribution over <code>p</code>.</figcaption>
</figure>

## Properties

Under regularity conditions (smooth log-likelihood, identifiable model, true parameter in interior), MLE is:

- **Consistent**: $\hat\theta \to \theta^*$ as $n \to \infty$.
- **Asymptotically normal**: $\sqrt{n}(\hat\theta - \theta^*) \to \mathcal{N}(0, I(\theta^*)^{-1})$ where $I(\theta)$ is the **Fisher information**.
- **Asymptotically efficient**: achieves the Cramér–Rao lower bound. No consistent estimator has lower asymptotic variance.

## Common cases

| Model | MLE solution |
|-------|-------------|
| Gaussian mean (known $\sigma^2$) | $\hat\mu = \bar x$ |
| Gaussian variance | $\hat\sigma^2 = \frac{1}{n} \sum (x_i - \bar x)^2$ (biased; sample-variance uses $n-1$) |
| Bernoulli ($p$) | $\hat p = \text{fraction of successes}$ |
| Categorical | empirical class frequencies |
| Linear regression with Gaussian noise | OLS: $\hat\beta = (X^\top X)^{-1} X^\top y$ |
| Logistic regression | no closed form; iterative (Newton, gradient methods) |

## Connection to cross-entropy and KL

For categorical $p_\theta(x)$ and empirical distribution $\hat p_\text{data}$, the negative log-likelihood divided by $n$ is

$$
-\tfrac{1}{n} \sum_i \log p_\theta(x_i) = H(\hat p_\text{data}, p_\theta) = H(\hat p_\text{data}) + \mathrm{KL}(\hat p_\text{data} \| p_\theta).
$$

Maximizing likelihood = minimizing cross-entropy = minimizing KL from the empirical distribution to the model.

## MLE vs MAP

MAP (maximum a posteriori) adds a prior: $\hat\theta_\text{MAP} = \arg\max_\theta \log p(\theta) + \sum \log p(x_i \mid \theta)$. MAP equals MLE when the prior is uniform (improper). Common choices:

- Gaussian prior $\Rightarrow$ L2 regularization on $\theta$.
- Laplace prior $\Rightarrow$ L1 regularization.

## Common pitfalls

- **Treating sample variance as MLE.** MLE for Gaussian variance divides by $n$ (biased); the sample variance divides by $n-1$ (unbiased). Different estimators.
- **Stopping at MLE without checking identifiability.** If two parameter values yield identical likelihoods, MLE is non-unique.
- **Trusting MLE on small samples.** Asymptotic guarantees can be misleading when $n$ is small relative to dimension; use cross-validation or Bayesian methods.
- **Forgetting that MLE on a misspecified model is still well-defined.** It converges to the parameter that minimizes KL to the (mismatched) true distribution within the model family.
