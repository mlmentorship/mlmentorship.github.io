---
title: "Maximum likelihood estimation"
description: "Compare parameter values using fixed observed data, derive a Bernoulli estimate, and distinguish likelihood, priors, and asymptotic guarantees."
date: "2025-09-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Maximum likelihood chooses a parameter that maximizes the probability or density assigned to the observed data.
The data stay fixed while candidate parameters vary.
For independent observations with a common model $p(x\mid\theta)$,

$$
\hat\theta_\text{MLE} = \arg\max_\theta \prod_{i=1}^{n} p(x_i \mid \theta) = \arg\max_\theta \sum_{i=1}^{n} \log p(x_i \mid \theta).
$$

The logarithm preserves the maximizer while turning products into sums.
For continuous observations, the factors are densities, not probabilities of exact values.
Dependent data require their joint or conditional likelihood rather than an unjustified independent product.

For seven successes and three failures under an independent Bernoulli model,

$$
L(p)=p^7(1-p)^3,\qquad
\ell'(p)=\frac7p-\frac3{1-p}.
$$

Setting the derivative to zero gives $7(1-p)=3p$, hence $\widehat p=0.7$.
The second derivative is negative for $0<p<1$, so this is the unique maximum for these mixed outcomes.
The diagram rescales likelihood by its peak only to make the shape readable.

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
	<figcaption><strong>Read it this way:</strong> keep the observations fixed while varying <code>p</code>. Seven successes contribute <code>p⁷</code>; three failures contribute <code>(1 − p)³</code>. Their product peaks at <code>0.7</code>. Relative likelihood is a parameter score, not a normalized parameter distribution.</figcaption>
</figure>

## Properties

Under appropriate identification, sampling, regularity, and existence conditions, MLE can be consistent and asymptotically normal:

$$
\sqrt n(\widehat\theta-\theta^*)\Rightarrow\mathcal N(0,I(\theta^*)^{-1}).
$$

Here $I$ is Fisher information per observation and must be nonsingular for this expression.
Efficiency statements compare regular estimators under their stated asymptotic framework; they do not rule out every pointwise superefficient estimator.
Small samples, boundary parameters, growing dimension, or misspecification require separate analysis.

## Common cases

| Model | MLE solution |
|-------|-------------|
| Gaussian mean (known $\sigma^2$) | $\hat\mu = \bar x$ |
| Gaussian variance | $\hat\sigma^2 = \frac{1}{n} \sum (x_i - \bar x)^2$ (biased; sample-variance uses $n-1$) |
| Bernoulli ($p$) | $\hat p = \text{fraction of successes}$ |
| Categorical | empirical class frequencies |
| Linear regression with independent equal-variance Gaussian noise | Least squares; the displayed inverse formula needs full column rank |
| Logistic regression | Iterative optimization; complete separation can prevent a finite optimum |

## Connection to cross-entropy and KL

For categorical $p_\theta(x)$ and empirical distribution $\hat p_\text{data}$, the negative log-likelihood divided by $n$ is

$$
-\tfrac{1}{n} \sum_i \log p_\theta(x_i) = H(\hat p_\text{data}, p_\theta) = H(\hat p_\text{data}) + \mathrm{KL}(\hat p_\text{data} \| p_\theta).
$$

The empirical entropy is constant as model parameters vary, so both objectives select the same optimum.
Gaussian regression produces squared-error minimization when its noise-variance and independence assumptions are appropriate.

## MLE vs MAP

### Add Prior Evidence

**New task:** Incorporate a declared prior rather than reporting only the likelihood maximizer.

**Picture and change:** A $\operatorname{Beta}(2,2)$ prior multiplies the Bernoulli likelihood by a factor proportional to $p(1-p)$.
The posterior is $\operatorname{Beta}(9,5)$.

**Check:** Its interior mode is $8/12=2/3$, not the MLE $0.7$; its mean is $9/14$.

**Use when:** Prior assumptions belong in the model.
MAP maximizes posterior density in the chosen parameterization; it is not generally invariant under reparameterization.
Gaussian and Laplace parameter priors produce quadratic and absolute-value penalties under their specified scales.

### Change the Observation Unit

**New task:** Fit a binomial count of seven successes rather than one particular ordered sequence.

**Picture and change:** Multiply by $\binom{10}{7}$ to account for compatible sequences.

**Check:** The factor equals $120$ and does not depend on $p$, so the maximizer remains $0.7$.

**Use when:** Aggregation changes the recorded observation but preserves the model's sufficient counts.
If dependence changes instead, multiplying independent factors need not be valid.

<!-- learning:core-end -->
<div data-learning-extension>

## Common pitfalls

- **Wrong variance denominator.** With an unknown Gaussian mean, the variance MLE divides residual squares by $n$.
	The unbiased sample-variance estimator uses $n-1$ under its assumptions.
- **Missing identification.** Parameters producing the same observation distribution need not have unique estimates.
- **Missing existence.** All-success Bernoulli data maximize at $p=1$ if the boundary is allowed.
	Restricting the parameter to $(0,1)$ leaves only a supremum.
- **Unqualified misspecification claims.** Suitable convergence conditions may select a KL-best approximation, not the true data-generating mechanism.

## Reconstruct

<details>
<summary>Is the plotted likelihood a posterior over the parameter?</summary>

No. It scores fixed data across parameter values and has not included a prior or normalized over parameters.
A constant rescaling preserves the likelihood maximizer; a nonconstant prior can change the posterior mode.

</details>

</div>
