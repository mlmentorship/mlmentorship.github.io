---
title: "Central limit theorem"
description: "Sums of many independent random variables become Gaussian. Why nearly every error bar in ML and statistics is computed from a normal distribution."
date: "2025-12-21"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

If $X_1, \dots, X_n$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2$, then as $n \to \infty$:

$$
\sqrt{n} \cdot (\bar X_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2).
$$

The standardized sample mean converges in distribution to a Gaussian, regardless of the original distribution's shape (as long as variance is finite).

**Learning objective:** distinguish the unchanged distribution of individual observations from the sampling distribution of the mean, which becomes more Gaussian and narrows at rate $1/\sqrt{n}$.

<!-- visual:clt-data-vs-sample-means -->
<figure class="learning-figure plot-panel" aria-labelledby="clt-sampling-title">
	<p class="visual-kicker">What becomes Gaussian?</p>
	<p class="visual-title" id="clt-sampling-title">Repeated sample means change shape; the original observations do not.</p>
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
	<figcaption><strong>Read it this way:</strong> compare rows at the dashed mean. The top distribution remains skewed because it describes individual observations. Each lower curve describes means from many hypothetical samples of the stated size: it becomes more bell-shaped while its spread shrinks from σ to σ/√n. The curves are schematic and assume independent observations with finite variance. Definitions checked against <a href="https://openstax.org/books/introductory-statistics-2e/pages/7-1-the-central-limit-theorem-for-sample-means-averages">OpenStax Introductory Statistics</a>; the graphic is original.</figcaption>
</figure>

The CLT is why we can:

- Build Gaussian-based confidence intervals for almost any estimator (sample mean, regression coefficients, A/B test deltas).
- Use $z$-tests and $t$-tests on data that isn't itself Gaussian.
- Trust that with enough samples, our reported metric ± std is approximately calibrated.

It also explains the prevalence of Gaussian assumptions in ML. Many quantities are sums or averages, and so naturally trend Gaussian.

## What "enough samples" means

The Berry–Esseen theorem bounds how fast convergence happens:

$$
\sup_x \big| F_n(x) - \Phi(x) \big| \le \frac{C \cdot \mathbb{E}[|X - \mu|^3]}{\sigma^3 \sqrt{n}}
$$

with $C \approx 0.4$. For symmetric, light-tailed distributions, $n = 30$ already gives an excellent Gaussian approximation. For heavy-tailed or skewed distributions you may need $n = 1000$ or more.

Heuristic check: plot a histogram of bootstrap means; if it looks Gaussian, the CLT has kicked in.

## Variants

- **Multivariate CLT**: $\sqrt{n}(\bar X_n - \mu) \to \mathcal{N}(0, \Sigma)$ for a vector-valued sum with covariance matrix $\Sigma$.
- **Lyapunov / Lindeberg CLT**: relaxes the i.i.d. assumption to independent (not identical) with mild moment conditions.
- **Martingale CLT**: extends to dependent data forming a martingale; used in online learning regret analysis.
- **CLT for U-statistics, M-estimators**: extends to functions of multiple samples.

## When CLT fails

- **Infinite variance** (e.g., Cauchy distribution): CLT does not apply; sample means do not concentrate. Use stable distributions instead.
- **Strong dependence**: highly correlated samples violate the i.i.d. assumption; effective sample size is much less than $n$.
- **Discrete distributions on small support**: CLT applies but the discrete approximation may be visibly bad until $n$ is large.

## Common pitfalls

- **Treating $n = 30$ as universally enough.** It is not for skewed or heavy-tailed data.
- **Using CLT on small sample sizes for inference.** Below $n \approx 30$, prefer the $t$-distribution (which uses the sample-estimated variance) rather than $z$.
- **Using parametric CLT confidence intervals on data that isn't independent.** A/B tests with user-level dependence (one user, multiple events) have effective $n$ much smaller than event count; cluster-bootstrap or use mixed-effects models.
- **Confusing "the mean is normally distributed" with "the data is normally distributed."** The CLT is about the *mean*, not individual samples.
