---
title: "Exponential family"
description: "A unified family of distributions (Gaussian, Bernoulli, Poisson, Beta, Gamma, etc.) with shared properties: sufficient statistics, conjugate priors, simple MLE."
date: "2025-12-03"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A distribution is in the **exponential family** if its density / mass function can be written as:

$$
p(x \mid \eta) = h(x) \exp\big( \eta^\top T(x) - A(\eta) \big)
$$

with **natural parameter** $\eta$, **sufficient statistic** $T(x)$, **base measure** $h(x)$, and **log-partition** $A(\eta)$ (which normalizes).

Most distributions you use day-to-day are exponential family: Gaussian, Bernoulli, categorical, Poisson, Beta, Gamma, Dirichlet, geometric, exponential. Recognizing them as such gives you free results:

- **MLE is closed-form** when the natural parameter is unconstrained: just match sample moments to model moments.
- **Conjugate priors** exist and are themselves exponential family.
- **Sufficient statistics** $T(x)$ contain all the data's information about $\theta$. You can summarize a dataset by $\sum_i T(x_i)$ and forget the rest.
- **Generalized linear models (GLMs)** are linear regression generalized to exponential-family responses.

## The canonical form

Given the form above:

- $T(x)$ are the **sufficient statistics** (Bernoulli: $T(x) = x$; Gaussian: $T(x) = (x, x^2)$).
- $\eta$ are the **natural parameters** (Bernoulli: $\eta = \log\frac{p}{1-p}$, the logit; Gaussian: $\eta = (\mu/\sigma^2, -1/(2\sigma^2))$).
- $A(\eta)$ is the **log-partition function**; its derivatives with respect to the natural parameter give the mean and covariance of $T(x)$:

$$
\nabla_\eta A(\eta) = \mathbb{E}_\eta[T(X)], \qquad \nabla_\eta^2 A(\eta) = \mathrm{Cov}_\eta(T(X)).
$$

This is why MLE via moment-matching works: the gradient of the log-likelihood is "data sufficient stat minus model expected sufficient stat."

<!-- visual:exponential-family-sufficient-stat-moment-match -->
<figure class="learning-figure" aria-labelledby="expfam-moment-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="expfam-moment-title">Why do sufficient statistics and moment matching come from the same factorization?</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 350" role="img" aria-labelledby="expfam-moment-svg-title expfam-moment-svg-desc">
			<title id="expfam-moment-svg-title">Six Bernoulli observations collapse to one sufficient statistic and one moment equation</title>
			<desc id="expfam-moment-svg-desc">The Bernoulli observations 1, 0, 1, 1, 0, and 1 map through T of x equals x and sum to the fixed-size sufficient statistic S equals 4. The parameter-dependent sample log-likelihood is 4 eta minus 6 A of eta, so the raw observations are no longer needed. Its score is 4 minus 6 sigmoid eta. Setting that score to zero matches the empirical mean S over n equals 4 over 6 to the model mean expected T of X equals p, yielding p hat equals 4 over 6.</desc>
			<defs>
				<marker id="expfam-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="4" y="4" width="352" height="342" rx="5"></rect>
			<text class="viz-axis-label" x="14" y="24">1 · REDUCE THE SAMPLE</text>
			<text class="viz-label" x="14" y="49">six Bernoulli observations</text>
			<circle class="viz-node viz-node--input" cx="35" cy="75" r="18"></circle>
			<circle class="viz-node viz-node--input" cx="91" cy="75" r="18"></circle>
			<circle class="viz-node viz-node--input" cx="147" cy="75" r="18"></circle>
			<circle class="viz-node viz-node--input" cx="203" cy="75" r="18"></circle>
			<circle class="viz-node viz-node--input" cx="259" cy="75" r="18"></circle>
			<circle class="viz-node viz-node--input" cx="315" cy="75" r="18"></circle>
			<text class="viz-node-label" x="35" y="80">1</text>
			<text class="viz-node-label" x="91" y="80">0</text>
			<text class="viz-node-label" x="147" y="80">1</text>
			<text class="viz-node-label" x="203" y="80">1</text>
			<text class="viz-node-label" x="259" y="80">0</text>
			<text class="viz-node-label" x="315" y="80">1</text>
			<path class="viz-forward" style="marker-end:url(#expfam-arrow)" d="M35 94L140 125M91 94L151 125M147 94L163 125M203 94L197 125M259 94L209 125M315 94L220 125"></path>
			<rect class="viz-node viz-node--focus" x="118" y="126" width="124" height="48" rx="24"></rect>
			<text class="viz-node-label" x="180" y="147">S = Σ T(xᵢ) = 4</text>
			<text class="viz-node-value" x="180" y="164">one number for any n</text>
			<text class="viz-axis-label" x="14" y="205">2 · WRITE THE SAMPLE LOG-LIKELIHOOD</text>
			<path class="viz-forward" style="marker-end:url(#expfam-arrow)" d="M180 174V217"></path>
			<rect class="viz-node" x="52" y="218" width="256" height="48" rx="5"></rect>
			<text class="viz-node-label" x="180" y="239">ℓ(η) = Sη − nA(η)</text>
			<text class="viz-node-value" x="180" y="256">here: 4η − 6A(η); raw xᵢ disappear</text>
			<text class="viz-axis-label" x="14" y="295">3 · SET THE SCORE TO ZERO</text>
			<path class="viz-forward" style="marker-end:url(#expfam-arrow)" d="M180 266V300"></path>
			<rect class="viz-node viz-node--output" x="32" y="301" width="296" height="34" rx="5"></rect>
			<rect class="viz-node viz-node--output" x="36" y="305" width="288" height="26" rx="3"></rect>
			<text class="viz-node-label" x="180" y="322">S/n = Eη[T(X)] = p̂ = 4/6</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> first replace the six observations by the single count S = 4. In the sample log-likelihood, every parameter-dependent data term now passes through S. Differentiating gives S − n∇A(η); when an interior MLE exists, setting this to zero matches the empirical statistic S/n to the model moment Eη[T(X)]. For Bernoulli data, that gives p̂ = 4/6. Original schematic checked against <a href="https://www.stat.berkeley.edu/~wfithian/courses/stat210a/exponential-families.html">Berkeley Stat 210A notes</a> and <a href="https://www.stat.umn.edu/geyer/5421/notes/expfam.html">Minnesota Stat 5421 notes</a>.</figcaption>
</figure>

## Common members

| Distribution | Sufficient stat $T(x)$ | Natural parameter $\eta$ |
|--------------|------------------------|--------------------------|
| Bernoulli($p$) | $x$ | $\log(p/(1-p))$ (logit) |
| Categorical($\pi$) | one-hot$(x)$ | $\log \pi$ (log-probabilities) |
| Gaussian($\mu, \sigma^2$) | $(x, x^2)$ | $(\mu/\sigma^2, -1/(2\sigma^2))$ |
| Poisson($\lambda$) | $x$ | $\log \lambda$ |
| Beta($\alpha, \beta$) | $(\log x, \log(1-x))$ | $(\alpha - 1, \beta - 1)$ |
| Gamma($\alpha, \beta$) | $(\log x, x)$ | $(\alpha - 1, -\beta)$ |

## Generalized linear models

A **GLM** combines a linear predictor $\eta = X \beta$ with an exponential-family response distribution. The link function maps the linear predictor to the natural parameter:

| Response | GLM | Link |
|----------|-----|------|
| Continuous (Gaussian) | linear regression | identity |
| Binary (Bernoulli) | logistic regression | logit |
| Count (Poisson) | Poisson regression | log |
| Categorical | multinomial logistic | softmax |
| Time-to-event (Exponential, Weibull) | survival models | log |

Logistic regression *is* a GLM with Bernoulli response and logit link. This unifies the entire family of "regression-style" classifiers.

## Properties to remember

- **Convexity**: $A(\eta)$ is convex in $\eta$. So negative log-likelihood is convex, and there is a unique MLE.
- **Sufficient statistics**: by Pitman-Koopman-Darmois theorem, exponential families are essentially the only distributions with finite-dimensional sufficient statistics independent of $n$.
- **Conjugacy**: exponential families have conjugate priors, also in the exponential family.
- **Maximum entropy**: the exponential family with sufficient statistics $T$ matching given moments is the maximum-entropy distribution under those constraints.

## Common pitfalls

- **Forgetting that the Cauchy distribution is not exponential family.** Heavy tails break the sufficient-statistic property.
- **Confusing "exponential" (the distribution) with "exponential family" (the class).** The Exp($\lambda$) distribution is one member.
- **Treating natural parameters as the same as canonical parameters.** A Gaussian's natural parameters are $(\mu/\sigma^2, -1/(2\sigma^2))$, not $(\mu, \sigma^2)$. Some software libraries default to one or the other; check.
