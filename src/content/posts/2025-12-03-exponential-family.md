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

For independent observations in a regular exponential family, an interior MLE satisfies that moment equation.
Such a finite interior optimum need not exist; boundary data, support restrictions, or nonidentifiability require separate treatment.
The [MLE lesson](/concepts/maximum-likelihood-estimation/) distinguishes a likelihood objective from a guarantee of existence or uniqueness.

<!-- visual:exponential-family-sufficient-stat-moment-match -->
<figure class="learning-figure" aria-labelledby="expfam-moment-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="expfam-moment-title">Compress sufficient counts, normalize the model, and match moments only where an interior fit exists</p><div class="visual-argument" data-visual-argument="exponential-family-sufficient-stat-moment-match" data-argument-kind="ledger" data-argument-composition="sufficient-compression"><svg viewBox="0 0 360 1675" role="img" aria-label="Compress sufficient counts, normalize the model, and match moments only where an interior fit exists. Natural parameters tilt weights; the log-partition function normalizes probabilities. Under this independent Bernoulli model, order is discarded but three successes and five trials remain. An interior maximum-likelihood fit matches the model's expected count to three. All-success data instead pushes the fit to probability one, requiring an unbounded natural parameter. Sufficient statistics depend on the assumed family; different models can require different summaries."><title>Compress sufficient counts, normalize the model, and match moments only where an interior fit exists</title><desc>Natural parameters tilt weights; the log-partition function normalizes probabilities. Under this independent Bernoulli model, order is discarded but three successes and five trials remain. An interior maximum-likelihood fit matches the model's expected count to three. All-success data instead pushes the fit to probability one, requiring an unbounded natural parameter. Sufficient statistics depend on the assumed family; different models can require different summaries.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Two orders, one sufficient count; fit</tspan><tspan x="20" dy="18">the model moment</tspan></text><circle cx="45" cy="103" r="19" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="45" y="109" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="45" dy="0">1</tspan></text><circle cx="111" cy="103" r="19" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="111" y="109" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="111" dy="0">0</tspan></text><circle cx="177" cy="103" r="19" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="177" y="109" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">1</tspan></text><circle cx="243" cy="103" r="19" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="243" y="109" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="243" dy="0">1</tspan></text><circle cx="309" cy="103" r="19" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="309" y="109" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="309" dy="0">0</tspan></text><circle cx="45" cy="204" r="19" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="45" y="210" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="45" dy="0">0</tspan></text><circle cx="111" cy="204" r="19" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="111" y="210" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="111" dy="0">1</tspan></text><circle cx="177" cy="204" r="19" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="177" y="210" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">1</tspan></text><circle cx="243" cy="204" r="19" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="243" y="210" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="243" dy="0">0</tspan></text><circle cx="309" cy="204" r="19" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="309" y="210" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="309" dy="0">1</tspan></text><path d="M105 240L160 300M151.64558641105825 296.8056653924634L160 300L157.5428195326642 291.3998683643247" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><path d="M255 240L200 300M202.4571804673358 291.3998683643247L200 300L208.35441358894175 296.8056653924634" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="180" y="337" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">3 successes; 5 trials</tspan></text><rect x="35" y="375" width="58" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><rect x="93" y="375" width="58" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><rect x="151" y="375" width="58" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><rect x="209" y="375" width="58" height="34" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></rect><rect x="267" y="375" width="58" height="34" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></rect><text x="20" y="458" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Three successes among five</tspan><tspan x="20" dy="23">trials</tspan></text><text x="20" y="530" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Order is discarded under this</tspan><tspan x="20" dy="18">independent Bernoulli model.</tspan></text><text x="20" y="635" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Natural parameters tilt weights. The</tspan><tspan x="20" dy="18">log-partition normalizes their</tspan><tspan x="20" dy="18">probabilities.</tspan></text><text x="20" y="745" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Model chance 20%; expected successes</tspan><tspan x="20" dy="18">1</tspan></text><rect x="20" y="790" width="320" height="36" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><rect x="20" y="790" width="64" height="36" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><path d="M212 780L212 836" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="878" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Model count does not match observed</tspan><tspan x="20" dy="18">three</tspan></text><text x="20" y="935" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Model chance 60%; expected successes</tspan><tspan x="20" dy="18">3</tspan></text><rect x="20" y="980" width="320" height="36" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><rect x="20" y="980" width="192" height="36" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><path d="M212 970L212 1026" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="1068" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Model count matches observed three</tspan></text><text x="20" y="1125" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Model chance 80%; expected successes</tspan><tspan x="20" dy="18">4</tspan></text><rect x="20" y="1170" width="320" height="36" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><rect x="20" y="1170" width="256" height="36" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><path d="M212 1160L212 1216" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="1258" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Model count does not match observed</tspan><tspan x="20" dy="18">three</tspan></text><text x="20" y="1350" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Common zero-to-five count scale; red</tspan><tspan x="20" dy="18">mark is the observed count.</tspan></text><text x="20" y="1455" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">All successes: probability one is a</tspan><tspan x="20" dy="18">boundary fit, not a finite interior</tspan><tspan x="20" dy="18">natural parameter.</tspan></text><text x="20" y="1575" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Different assumed families can</tspan><tspan x="20" dy="18">require different sufficient</tspan><tspan x="20" dy="18">statistics.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> first replace the six observations by the single count S = 4. In the sample log-likelihood, every parameter-dependent data term now passes through S. Differentiating gives S − n∇A(η); when an interior MLE exists, setting this to zero matches the empirical statistic S/n to the model moment Eη[T(X)]. For Bernoulli data, that gives p̂ = 4/6. Original schematic checked against <a href="https://www.stat.berkeley.edu/~wfithian/courses/stat210a/exponential-families.html">Berkeley Stat 210A notes</a> and <a href="https://www.stat.umn.edu/geyer/5421/notes/expfam.html">Minnesota Stat 5421 notes</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="exponential-family-sufficient-stat-moment-match"><strong>Check:</strong> What must remain known alongside S if samples can have different lengths?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="exponential-family-sufficient-stat-moment-match">Natural parameters tilt weights; the log-partition function normalizes probabilities. Under this independent Bernoulli model, order is discarded but three successes and five trials remain. An interior maximum-likelihood fit matches the model's expected count to three. All-success data instead pushes the fit to probability one, requiring an unbounded natural parameter. Sufficient statistics depend on the assumed family; different models can require different summaries. <strong>Does all-success data give a finite natural parameter?</strong></span></figcaption>
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
