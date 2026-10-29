---
title: "Central limit theorem: what becomes Gaussian?"
description: "Calculate the spread of repeated sample means, compare exact and Gaussian tails, and check what changes under dependence."
date: "2025-12-21"
draft: false
tags: ["concepts"]
category: "concepts"
---

The central limit theorem approximates standardized averages without changing the distribution of individual observations.
Which distribution narrows when we average fresh samples of skewed waiting times?

## Summary

Suppose $X_1,\ldots,X_n$ are independent, identically distributed observations with mean $\mu$ and finite variance $0<\sigma^2<\infty$.
Each repetition draws a fresh sample and computes $\bar X_n=n^{-1}\sum_iX_i$.
The population law stays unchanged across repetitions.

Linearity and independence give exact identities:

$$
\mathbb E[\bar X_n]=\mu,\qquad
\operatorname{Var}(\bar X_n)=\frac{1}{n^2}\sum_i\sigma^2=\frac{\sigma^2}{n}.
$$

The standard error, $\sigma/\sqrt n$, describes variation across repeated means, not individual observations.
The theorem separately supplies a limiting shape:

$$
Z_n=\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\xrightarrow{d}\mathcal N(0,1).
$$

This is convergence of distribution functions at continuity points, not exact finite-sample normality.

<!-- visual:clt-data-vs-sample-means -->
<figure class="learning-figure plot-panel" aria-labelledby="clt-sampling-title">
	<p class="visual-kicker">What becomes Gaussian?</p>
	<p class="visual-title" id="clt-sampling-title">Averaging changes the mean's distribution, not the original observations</p>
	<svg viewBox="0 0 360 310" role="img" aria-labelledby="clt-sampling-svg-title clt-sampling-svg-desc">
		<title id="clt-sampling-svg-title">A skewed population compared with sampling distributions of its mean</title>
		<desc id="clt-sampling-svg-desc">Three aligned schematic density plots share the same population mean. The top row shows a right-skewed distribution for one observation X. The middle row shows the less skewed distribution across means of repeated samples of size four. The bottom row shows an approximately Gaussian and narrower distribution across means of repeated samples of size twenty-five. A vertical dashed line marks the unchanged mean mu, while the row labels state that standard error shrinks as sigma divided by the square root of n.</desc>
		<rect class="viz-plot-bg" x="104" y="28" width="236" height="236" rx="3"></rect>
		<path class="viz-operating-guide" d="M224 28V264"></path>
		<text class="viz-callout" x="224" y="18" text-anchor="middle">same center μ</text>
		<text class="viz-axis-label" x="8" y="58">ONE OBSERVATION</text>
		<text class="viz-callout" x="8" y="75">X</text>
		<text class="viz-label" x="8" y="91">still skewed</text>
		<path class="viz-axis" d="M110 96H334"></path>
		<path class="viz-roc-curve" d="M112 96C119 95 120 49 135 43C153 36 169 62 184 74C201 87 230 93 268 95C292 96 314 96 332 96"></path>
		<path class="viz-gridline" d="M104 112H340"></path>
		<text class="viz-axis-label" x="8" y="145">REPEATED MEANS</text>
		<text class="viz-callout" x="8" y="162">X̄, n = 4</text>
		<text class="viz-label" x="8" y="178">SE = σ / √4</text>
		<path class="viz-axis" d="M110 188H334"></path>
		<path class="viz-roc-curve" d="M134 188C151 186 167 153 188 139C206 127 225 128 242 140C260 153 270 178 300 186C310 188 322 188 332 188"></path>
		<path class="viz-gridline" d="M104 204H340"></path>
		<text class="viz-axis-label" x="8" y="237">REPEATED MEANS</text>
		<text class="viz-callout" x="8" y="254">X̄, n = 25</text>
		<text class="viz-label" x="8" y="270">SE = σ / √25</text>
		<path class="viz-axis" d="M110 280H334"></path>
		<path class="viz-roc-curve" d="M171 280C188 279 195 264 204 240C211 221 218 214 224 214C230 214 237 221 244 240C253 264 260 279 277 280"></path>
		<text class="viz-label" x="224" y="301" text-anchor="middle">possible values of X or X̄</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> compare rows at the dashed mean. The top row describes unchanged individual observations. Each lower curve describes means across fresh independent samples. Their standard error is σ/√n. These are schematic curves, not exact finite-sample densities. The reference is <a href="https://openstax.org/books/introductory-statistics-2e/pages/7-1-the-central-limit-theorem-for-sample-means-averages">OpenStax Introductory Statistics</a>; the graphic is original. <span class="visual-reader-check" data-reader-check="clt-data-vs-sample-means"><strong>Check:</strong> Does averaging make the individual observations Gaussian, or only change the distribution of their mean?</span></figcaption>
</figure>

## What "enough samples" means

Use hypothetical waiting times with an exponential distribution of rate $0.5$ per minute.
Then $\mu=2$ minutes and $\sigma^2=4$ square minutes.

| Independent observations | Mean of repeated averages | Variance | Standard error |
| --- | --- | --- | --- |
| $4$ | $2$ | $1$ | $1$ |
| $25$ | $2$ | $0.16$ | $0.4$ |

At $n=25$, the threshold $\bar X_n>2.8$ standardizes to $(2.8-2)/0.4=2$.
The Gaussian upper tail is about $0.02275$.
The exact sum has a Gamma distribution with shape $25$ and scale $2$:

$$
\Pr(\bar X_{25}>2.8)
=e^{-35}\sum_{j=0}^{24}\frac{35^j}{j!}
\approx0.032374.
$$

The exact variance calculation did not require a Gaussian approximation; the tail calculation did.
No universal sample count makes every tail approximation accurate.

### Change the precision target

To halve standard error from $0.4$ to $0.2$, solve $2/\sqrt n=0.2$.
The answer is $n=100$, four times the observations.
Doubling observations only divides standard error by $\sqrt2$.
Use this calculation when precision, rather than sample count, is the design target.

### Change the independent unit

Repeat each of five independent waiting times five times, producing 25 rows.
Their average equals the average of the original five times.
Its variance is $4/5=0.8$, with standard error $\sqrt{0.8}\approx0.894427$, not $0.4$.
Covariances between repeated rows prevent the independent-row variance calculation.
Use the actual independent units when designing clustered or repeated-measurement inference.

## When CLT fails

Averages of independent standard Cauchy observations remain standard Cauchy.
Their variance is undefined, so this theorem's premises fail.

Finite variance alone does not make $n=30$ sufficient.
For Bernoulli probability $0.001$, all 30 observations are zero with probability $0.999^{30}\approx0.970431$.
That discrete mass defeats a universal bell-shape rule.

**Reconstruct:** independent observations have variance $9$. Which sample count gives standard error $0.3$? Does it change the observation law?

<details><summary>Answer</summary>

Solve $3/\sqrt n=0.3$ to obtain $n=100$.
The observation law stays unchanged; only the standardized mean has the stated Gaussian limit.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Variants

For iid observations with finite third absolute centered moment $\rho$, Berry-Esseen bounds approximation error:

$$
\sup_x|\Pr(Z_n\le x)-\Phi(x)|
\le C\frac{\rho}{\sigma^3\sqrt n}.
$$

Here $\Phi$ is the standard normal distribution function and $C$ is a universal constant.
For this iid bound, $C=0.4748$ is valid by [Shevtsova's upper estimate](https://arxiv.org/abs/1111.6554).
The third-moment condition is stronger than finite variance.
Large standardized third moments make this bound less informative.

Multivariate CLTs replace scalar variance with covariance for iid vectors with finite second moments.
Lindeberg conditions handle suitable independent nonidentical triangular arrays.
Martingale CLTs require conditional-variance convergence and controls on large increments; dependence alone does not establish them.
Estimator-specific CLTs need their own expansions and regularity conditions.
Neither arbitrary estimators nor degenerate U-statistics automatically have this mean's normalization or Gaussian limit.

The law of large numbers concerns concentration toward $\mu$, not the standardized fluctuation's shape.
These are different conclusions even when both theorems apply.

## Common pitfalls

Student's exact small-sample $t$ law requires iid Gaussian observations and their sample variance.
Consistent variance estimation instead permits asymptotic studentization under appropriate conditions.
It is not a generic small-sample repair.

A bootstrap histogram describes resampling from observed data.
It cannot establish coverage when the sample missed rare outcomes or used the wrong sampling unit.
Metric uncertainty requires an estimand, independent units, a suitable scale estimate, and a justified approximation.
Writing a metric plus or minus an unspecified standard deviation supplies none of those guarantees.

</div>
