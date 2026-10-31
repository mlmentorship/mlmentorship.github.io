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
	<p class="visual-title" id="clt-sampling-title">Averaging changes the mean's distribution, not the original observations</p><div class="visual-argument" data-visual-argument="clt-data-vs-sample-means" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 439" role="img" aria-label="Averages concentrate; the underlying observations do not change. For independent exponential observations, a sample mean has variance 1/n. Averaging does not turn the original skewed population into a Gaussian."><title>Averages concentrate; the underlying observations do not change</title><desc>For independent exponential observations, a sample mean has variance 1/n. Averaging does not turn the original skewed population into a Gaussian.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">0</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">4</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">1.3</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="1" d="M118.5 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><text x="118.5" y="245" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="118.5" dy="0">1</tspan></text><path data-proof-series="Original exponential density" d="M48 89.23076923076925L52.40625 97.1536764013147L56.8125 104.59655889278368L61.21875 111.58849993025534L65.625 118.15682067527783L70.03125 124.3271869839007L74.4375 130.1237096965652L78.84375 135.5690388517372L83.25 140.68445219142487L87.65625 145.48993830441776L92.0625 150.00427473213205L96.46875 154.2451013422616L100.875 158.22898925694423L105.28125 161.97150560478187L109.6875 165.48727434973353L114.09375 168.79003343457242L118.5 171.89268846219602L122.90625 174.80736312454948L127.3125 177.5454465762158L131.71875 180.11763793778456L136.125 182.5339881028982L140.53125 184.80393901233748L144.9375 186.93636054861008L149.34375 188.93958519520933L153.75 190.82144059597456L158.15625 192.5892801417795L162.5625 194.25001170406694L166.96875 195.81012462750635L171.375 197.2757150872495L175.78125 198.65250990986698L180.1875 199.9458889510478L184.59375 201.16090611750226L189 202.30230911521218L193.40625 203.37455800119534L197.8125 204.38184261127512L202.21875 205.32809893195505L206.625 206.21702448037158L211.03125 207.0520927524227L215.4375 207.83656679552863L219.84375 208.5735119590604L224.25 209.2658078722594L228.65625 209.9161596964509L233.0625 210.52710869552095L237.46875 211.1010421659602L241.875 211.6402027652767L246.28125 212.14669727522897L250.6875 212.62250483412143L255.09375 213.06948467033038L259.5 213.48938336727934L263.90625 213.88384168825152L268.3125 214.25440098770827L272.71875 214.60250923416493L277.125 214.92952666815944L281.53125 215.2367311174216L285.9375 215.5253229900129L290.34375 215.79642996494698L294.75 216.0511113986199L299.15625 216.2903624642683L303.5625 216.51511804063043L307.96875 216.7262563650052L312.375 216.92460246498342L316.78125 217.11093138226107L321.1875 217.28597120113156L325.59375 217.45040589349102L330 217.60487799147322" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="1" data-proof-y="0.36787944117144233"><circle cx="118.5" cy="171.89268846219602" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Mean of 10; normal approximation" d="M48 218.88841359813196L52.40625 217.9634505373582L56.8125 216.41175514050258L61.21875 213.91998638141072L65.625 210.09253824493032L70.03125 204.47414131869138L74.4375 196.60170431176292L78.84375 186.088406879677L83.25 172.73414345601412L87.65625 156.64480324287015L92.0625 138.33192342795195L96.46875 118.75868385137939L100.875 99.30240700296308L105.28125 81.61952787369398L109.6875 67.42394609300604L114.09375 58.21683744749072L118.5 55.02595048329721L122.90625 58.21683744749072L127.3125 67.42394609300604L131.71875 81.61952787369398L136.125 99.30240700296308L140.53125 118.75868385137939L144.9375 138.33192342795195L149.34375 156.64480324287015L153.75 172.73414345601412L158.15625 186.088406879677L162.5625 196.60170431176292L166.96875 204.47414131869138L171.375 210.09253824493032L175.78125 213.91998638141072L180.1875 216.41175514050258L184.59375 217.9634505373582L189 218.88841359813196L193.40625 219.4165188099101L197.8125 219.70545892564593L202.21875 219.85701178491695L206.625 219.93324404779887L211.03125 219.97002804189017L215.4375 219.98705876836675L219.84375 219.99462632401736L224.25 219.99785413343275L228.65625 219.99917592009785L233.0625 219.99969565140339L237.46875 219.99989190424077L241.875 219.99996307832623L246.28125 219.99998787199195L250.6875 219.99999616881658L255.09375 219.9999988361104L259.5 219.99999965996315L263.90625 219.99999990446213L268.3125 219.9999999741857L272.71875 219.9999999932922L277.125 219.99999999832374L281.53125 219.99999999959715L285.9375 219.9999999999069L290.34375 219.9999999999793L294.75 219.99999999999557L299.15625 219.9999999999991L303.5625 219.99999999999983L307.96875 219.99999999999997L312.375 220L316.78125 220L321.1875 220L325.59375 220L330 220" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="1" data-proof-y="1.2615662610100802"><circle cx="118.5" cy="55.02595048329721" r="5" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">observed value or sample mean</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">probability density</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Original exponential density</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">0.3679 at 1</tspan></text><path d="M20 368L48 368" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="372" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Mean of 10; normal approximation</tspan></text><text x="60" y="395" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="60" dy="0">1.262 at 1</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> compare rows at the dashed mean. The top row describes unchanged individual observations. Each lower curve describes means across fresh independent samples. Their standard error is σ/√n. These are schematic curves, not exact finite-sample densities. The reference is <a href="https://openstax.org/books/introductory-statistics-2e/pages/7-1-the-central-limit-theorem-for-sample-means-averages">OpenStax Introductory Statistics</a>; the graphic is original.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="clt-data-vs-sample-means"><strong>Check:</strong> Does averaging make the individual observations Gaussian, or only change the distribution of their mean?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="clt-data-vs-sample-means">For independent exponential observations, a sample mean has variance 1/n. Averaging does not turn the original skewed population into a Gaussian.</span></figcaption>
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
