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
	<p class="visual-title" id="clt-sampling-title">Averaging changes the mean's distribution, not the original observations</p><div class="visual-argument" data-visual-argument="clt-data-vs-sample-means" data-argument-kind="curve" data-argument-composition="average-distribution"><svg viewBox="0 0 360 555" role="img" aria-label="Averages concentrate; the underlying observations do not change. The original exponential population stays unchanged. Ten independent observations produce one mean. Here, its exact Gamma density is narrower. Its population variance is one tenth."><title>Averages concentrate; the underlying observations do not change</title><desc>The original exponential population stays unchanged. Ten independent observations produce one mean. Here, its exact Gamma density is narrower. Its population variance is one tenth.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">The population does not become an</tspan><tspan x="20" dy="18">average</tspan></text><text x="20" y="77" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Original exponential population</tspan></text><path d="M35 180L35 108L38.625 111.5114814359486L42.25 114.85170590141091L45.875 118.02902569739584L49.5 121.05138577838531L53.125 123.92634361885885L56.75 126.66108811091631L60.375 129.26245754025263L64 131.73695668543397L67.625 134.09077308323234L71.25 136.3297925006904L74.875 138.45961365260496L78.5 140.4855622012301L82.125 142.41270407320684L85.75 144.24585812701852L89.375 145.98960820264693L93 147.64831458356005L96.625 149.22612489969168L100.25 150.72698449867687L103.875 152.1546463112759L107.5 153.51268023565615L111.125 154.80448206399683L114.75 156.03328197373827L118.375 157.20215260470817L122 158.31401674232146L125.625 159.37165462606632L129.25 160.3777109015511L132.875 161.33470123349582L136.5 162.24501859620432L140.125 163.11093925724657L143.75 163.93462846931305L147.375 164.7181458844745L151 165.4634507043848L154.625 166.17240657930572L158.25 166.84678626820312L161.875 167.48827607156795L165.5 168.09848004804576L169.125 168.6789240254188L172.75 169.23105941597026L176.375 169.75626684577102L180 170.2558596069639L183.625 170.7310869416781L187.25 171.1831371657853L190.875 171.61314064030822L194.5 172.02217259791195L198.125 172.41125583154576L201.75 172.78136325195814L205.375 173.13342032048044L209 173.4683073631623L212.625 173.7868617720453L216.25 174.0898800990793L219.875 174.37812004791698L223.5 174.65230236856797L227.125 174.91311265964907L230.75 175.161203082738L234.375 175.39719399311704L238 175.6216754909843L241.625 175.83520889701163L245.25 176.03832815593867L248.875 176.23154117171288L252.5 176.4153310775138L256.125 176.59015744383785L259.75 176.75645742766383L263.375 176.9146468655731L267 177.06512131355763L270.625 177.208257036116L274.25 177.34441194711073L277.875 177.47392650473915L281.5 177.59712456285652L285.125 177.7143141807791L288.75 177.82578839359306L292.375 177.93182594489477L296 178.03269198379493L299.625 178.12863872792963L303.25 178.21990609413555L306.875 178.30672229836733L310.5 178.3893044263561L314.125 178.46785897643684L317.75 178.54258237590207L321.375 178.61366147217214L325 178.68127400001114L325 180Z" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></path><circle cx="37.175" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M37.175 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="45.15" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M45.15 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="56.025" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M56.025 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="67.625" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M67.625 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="79.225" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M79.225 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="99.525" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M99.525 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="114.75" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M114.75 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="137.95" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M137.95 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="187.25" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M187.25 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="250.32500000000002" cy="235" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M250.32500000000002 245L107.5 291" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="107.5" cy="304" r="7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="340" y="284" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--viz-input-stroke)"><tspan x="340" dy="0">Ten draws</tspan></text><text x="340" y="320" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">One mean</tspan></text><text x="20" y="357" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Distribution of means of ten</tspan></text><path d="M35 466L35 466L38.625 465.9999976495433L42.25 465.9992700804739L45.875 465.98298041084524L49.5 465.8625165376644L53.125 465.37871146865194L56.75 464.05563716927645L60.375 461.27772607013986L64 456.4735419824438L67.625 449.3215027004854L71.25 439.8887842607365L74.875 428.65658607324565L78.5 416.4362879051948L82.125 404.21606840869026L85.75 392.9886379595745L89.375 383.6028444076734L93 376.66461955161793L96.625 372.4945986201006L100.25 371.1359391931436L103.875 372.39817142156716L107.5 375.92077428078403L111.125 381.24192292832134L114.75 387.86163330529234L118.375 395.2928251873954L122 403.09764646980443L125.625 410.909266239377L129.25 418.4411472636824L132.875 425.48667627282305L136.5 431.91219490547934L140.125 437.64618534745216L143.75 442.66683960178943L147.375 446.98964278438933L151 450.6560350771789L154.625 453.7237419148886L158.25 456.2589995456237L161.875 458.3306515778266L165.5 460.005936558571L169.125 461.3477065145278L172.75 462.41279070723556L176.375 463.25122894677827L180 463.90612965339574L183.625 464.41394832090987L187.25 464.8050245139117L190.875 465.1042553109503L194.5 465.33181766616394L198.125 465.5038805155394L201.75 465.63326960693945L205.375 465.730064564868L209 465.8021194661818L212.625 465.85550613225996L216.25 465.8948843224468L219.875 465.9238058201092L223.5 465.94496067921807L227.125 465.9603741529811L230.75 465.9715624400469L234.375 465.9796546364009L238 465.9854873656795L241.625 465.98967760510726L245.25 465.9926783077658L248.875 465.9948205888043L252.5 465.9963455140298L256.125 465.99742790906004L259.75 465.99819409138803L263.375 465.99873500657645L267 465.999115911284L270.625 465.99938347733007L274.25 465.9995709805126L277.875 465.9997020745643L281.5 465.9997935250678L285.125 465.9998571824122L288.75 465.99990140043565L292.375 465.99993205296136L296 465.99995325979404L299.625 465.99996790357557L303.25 465.99997799663265L306.875 465.99998494059537L310.5 465.9999897096174L314.125 465.99999297933135L317.75 465.99999521737715L321.375 465.99999674679606L325 465.9999977903163L325 466Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></path><text x="20" y="515" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same center. One tenth the</tspan><tspan x="20" dy="23">variance.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> compare rows at the dashed mean. The top row describes unchanged individual observations. Each lower curve describes means across fresh independent samples. Their standard error is σ/√n. These are schematic curves, not exact finite-sample densities. The reference is <a href="https://openstax.org/books/introductory-statistics-2e/pages/7-1-the-central-limit-theorem-for-sample-means-averages">OpenStax Introductory Statistics</a>; the graphic is original.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="clt-data-vs-sample-means"><strong>Check:</strong> Does averaging make the individual observations Gaussian, or only change the distribution of their mean?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="clt-data-vs-sample-means">The original exponential population stays unchanged. Ten independent observations produce one mean. Here, its exact Gamma density is narrower. Its population variance is one tenth. <strong>Does collecting means alter the distribution of an individual future observation?</strong></span></figcaption>
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
