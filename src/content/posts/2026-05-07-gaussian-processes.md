---
title: "Gaussian processes"
description: "A distribution over functions defined by a mean and covariance kernel. Predicts a mean and kernel-conditional uncertainty. Beautiful theory, brutal scaling."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **Gaussian process** (GP) places a prior over functions $f: \mathcal{X} \to \mathbb{R}$ such that any finite collection of values $\{f(x_1), \dots, f(x_n)\}$ is jointly Gaussian. The GP is fully specified by a mean function $m(x)$ (usually 0) and a covariance kernel $k(x, x')$.

GPs are the canonical Bayesian regression model. They give a closed-form posterior over functions, with a predictive mean and a predictive variance per query point. Under a suitable, well-specified kernel, that variance can be calibrated and returns toward the prior variance as kernel similarity to the observations falls. This makes GPs a standard tool for **Bayesian optimization**, **active learning**, and settings where uncertainty quantification matters as much as the prediction.

The cost is brutal scaling: $O(n^3)$ exact inference. GPs are practical up to a few thousand training points without approximation; modern variants (sparse, structured kernel, deep kernel) push that to millions.

## The mechanism

For training inputs $X$, training targets $\mathbf{y}$, test point $x_*$, prior covariance kernel $k$, and noise variance $\sigma^2$:

The joint distribution is

$$
\begin{bmatrix} \mathbf{y} \\ f(x_*) \end{bmatrix} \sim \mathcal{N}\!\left(\mathbf{0}, \begin{bmatrix} K + \sigma^2 I & k_* \\ k_*^\top & k_{**} \end{bmatrix}\right),
$$

where $K_{ij} = k(x_i, x_j)$, $k_*$ is the vector of $k(x_*, x_i)$, and $k_{**} = k(x_*, x_*)$.

Conditioning on training data gives the Gaussian posterior:

$$
\begin{aligned}
\mu(x_*) &= k_*^\top (K + \sigma^2 I)^{-1} \mathbf{y}, \\
\Sigma(x_*) &= k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*.
\end{aligned}
$$

<!-- visual:gp-conditioning-local-uncertainty -->
<figure class="learning-figure" aria-labelledby="gp-conditioning-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="gp-conditioning-title">Conditioning narrows uncertainty where observed points are informative</p><div class="visual-argument" data-visual-argument="gp-conditioning-local-uncertainty" data-argument-kind="curve" data-argument-composition="uncertainty-pin"><svg viewBox="0 0 360 639" role="img" aria-label="An exact observation removes uncertainty locally. The noiseless observation pins uncertainty at zero. The symmetric band shows plus or minus one posterior standard deviation. This unit RBF example reopens away from the observed input."><title>An exact observation removes uncertainty locally</title><desc>The noiseless observation pins uncertainty at zero. The symmetric band shows plus or minus one posterior standard deviation. This unit RBF example reopens away from the observed input.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">An observation pins the band only</tspan><tspan x="20" dy="18">locally</tspan></text><text x="20" y="67" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Prior: equal uncertainty across</tspan><tspan x="20" dy="18">inputs</tspan></text><path d="M35 111L325 111L325 245L35 245Z" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></path><path d="M35 178L325 178" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="335" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Posterior: zero width at the observed</tspan><tspan x="20" dy="18">input</tspan></text><path d="M35 379.00413435599586L39.53125 379.0071926907268L44.0625 379.0122955355091L48.59375 379.02065284934986L53.125 379.0340874790762L57.65625 379.05528418095673L62.1875 379.08810701887137L66.71875 379.1379880306955L71.25 379.2123820798644L75.78125 379.3212716682695L80.3125 379.47769177439903L84.84375 379.6982299217157L89.375 380.0034429106902L93.90625 380.418121732598L98.4375 380.97133290366196L102.96875 381.69616993976183L107.5 382.6291637979915L112.03125 383.8093251040566L116.5625 385.2768215857834L121.09375 387.07132796566316L125.625 389.23011881422985L130.15625 391.78600393681216L134.6875 394.76522786185916L139.21875 398.18546794431677L143.75 402.0540684501638L148.28125 406.36664053558655L152.8125 411.1061407695938L157.34375 416.24251487807123L161.875 421.73296036933846L166.40625 427.52282379341125L170.9375 433.5471081597925L175.46875 439.7325263365537L180 446L184.53125 439.7325263365537L189.0625 433.5471081597925L193.59375 427.52282379341125L198.125 421.73296036933846L202.65625 416.24251487807123L207.1875 411.1061407695938L211.71875 406.36664053558655L216.25 402.0540684501638L220.78125 398.18546794431677L225.3125 394.76522786185916L229.84375 391.78600393681216L234.375 389.23011881422985L238.90625 387.07132796566316L243.4375 385.2768215857834L247.96875 383.8093251040566L252.5 382.6291637979915L257.03125 381.69616993976183L261.5625 380.97133290366196L266.09375 380.418121732598L270.625 380.0034429106902L275.15625 379.6982299217157L279.6875 379.47769177439903L284.21875 379.3212716682695L288.75 379.2123820798644L293.28125 379.1379880306955L297.8125 379.08810701887137L302.34375 379.05528418095673L306.875 379.0340874790762L311.40625 379.02065284934986L315.9375 379.0122955355091L320.46875 379.0071926907268L325 379.00413435599586L325 512.9958656440042L320.46875 512.9928073092732L315.9375 512.9877044644909L311.40625 512.9793471506501L306.875 512.9659125209238L302.34375 512.9447158190433L297.8125 512.9118929811286L293.28125 512.8620119693045L288.75 512.7876179201356L284.21875 512.6787283317304L279.6875 512.522308225601L275.15625 512.3017700782843L270.625 511.9965570893098L266.09375 511.581878267402L261.5625 511.02866709633804L257.03125 510.30383006023817L252.5 509.3708362020085L247.96875 508.1906748959434L243.4375 506.7231784142166L238.90625 504.92867203433684L234.375 502.76988118577015L229.84375 500.21399606318784L225.3125 497.23477213814084L220.78125 493.81453205568323L216.25 489.9459315498362L211.71875 485.63335946441345L207.1875 480.8938592304062L202.65625 475.75748512192877L198.125 470.26703963066154L193.59375 464.47717620658875L189.0625 458.4528918402075L184.53125 452.2674736634463L180 446L175.46875 452.2674736634463L170.9375 458.4528918402075L166.40625 464.47717620658875L161.875 470.26703963066154L157.34375 475.75748512192877L152.8125 480.8938592304062L148.28125 485.63335946441345L143.75 489.9459315498362L139.21875 493.81453205568323L134.6875 497.23477213814084L130.15625 500.21399606318784L125.625 502.76988118577015L121.09375 504.92867203433684L116.5625 506.7231784142166L112.03125 508.1906748959434L107.5 509.3708362020085L102.96875 510.30383006023817L98.4375 511.02866709633804L93.90625 511.581878267402L89.375 511.9965570893098L84.84375 512.3017700782843L80.3125 512.522308225601L75.78125 512.6787283317304L71.25 512.7876179201356L66.71875 512.8620119693045L62.1875 512.9118929811286L57.65625 512.9447158190433L53.125 512.9659125209238L48.59375 512.9793471506501L44.0625 512.9877044644909L39.53125 512.9928073092732L35 512.9958656440042Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></path><path d="M35 446L325 446" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="180" cy="446" r="8" style="fill:var(--viz-focus-stroke);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><text x="20" y="590" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">The envelope reopens away from the</tspan><tspan x="20" dy="18">observation.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 270" role="img" aria-labelledby="gp-posterior-title gp-posterior-desc">
			<title id="gp-posterior-title">Gaussian process posterior mean and uncertainty around three observations</title>
			<desc id="gp-posterior-desc">A horizontal input axis contains three observed points. A solid posterior mean bends through their neighborhood. A shaded uncertainty band is narrow near each observation and broad at the left and right edges, where there is little kernel similarity to the training inputs. Labels connect the narrow regions to a large covariance correction subtracted from prior variance and the broad regions to a near-zero correction.</desc>
			<path d="M30 82C55 87 76 90 103 92C126 70 148 73 174 87C200 113 221 128 246 137C272 130 298 100 334 77L334 173C298 162 272 160 246 153C221 148 200 143 174 103C148 93 126 90 103 108C76 150 55 153 30 178Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></path>
			<path d="M30 130C55 120 76 111 103 100C126 80 148 83 174 95C200 126 221 138 246 145C272 140 298 131 334 125" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3"></path>
			<path class="viz-axis" d="M28 205H340"></path>
			<path class="viz-axis" d="M30 48V205"></path>
			<path d="M103 100V205M174 95V205M246 145V205" style="fill:none;stroke:var(--viz-edge);stroke-width:1;stroke-dasharray:3 3"></path>
			<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"><circle cx="103" cy="100" r="5"></circle><circle cx="174" cy="95" r="5"></circle><circle cx="246" cy="145" r="5"></circle></g>
			<text class="viz-axis-label" x="30" y="38">OUTPUT f(x)</text>
			<text class="viz-axis-label" x="340" y="221" text-anchor="end">INPUT x</text>
			<text class="viz-callout" x="185" y="120">posterior mean &mu;(x)</text>
			<text class="viz-callout" x="220" y="82">posterior uncertainty</text>
			<text class="viz-axis-label" x="174" y="238" text-anchor="middle">OBSERVED INPUTS</text>
			<path d="M77 62H119M77 57V67M119 57V67" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5"></path>
			<text class="viz-label" x="98" y="53" text-anchor="middle">near X</text>
			<text class="viz-label" x="98" y="252" text-anchor="middle">large k* &rarr; large subtraction &rarr; narrow</text>
			<path d="M291 55H334M291 50V60M334 50V60" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
			<text class="viz-label" x="313" y="46" text-anchor="middle">far from X</text>
			<text class="viz-label" x="278" y="252" text-anchor="middle">k* &asymp; 0 &rarr; prior width remains</text>
		</svg>
	</div>
	<div class="visual-derivation-caption">start at a query location and ask how similar it is to the observed inputs. Nearby, the covariance vector <var>k</var><sub>*</sub> is large: observations pull the mean and the subtracted correction in &Sigma;(x<sub>*</sub>) removes much of the prior uncertainty. Far away, <var>k</var><sub>*</sub> approaches zero, so the mean relaxes toward the prior and the uncertainty band widens toward the prior variance. This behavior is conditional on the chosen kernel; it is not a universal guarantee of calibration. Original schematic checked against <a href="https://gaussianprocess.org/gpml/">Rasmussen and Williams (2006)</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="gp-conditioning-local-uncertainty"><strong>Check:</strong> Which covariance term makes the posterior band narrower at the marked query? What uncertainty does that band represent?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="gp-conditioning-local-uncertainty">The noiseless observation pins uncertainty at zero. The symmetric band shows plus or minus one posterior standard deviation. This unit RBF example reopens away from the observed input. <br><strong>Does one noiseless observation eliminate uncertainty everywhere?</strong></span></figcaption>
</figure>

Two matrix-vector products against $(K + \sigma^2 I)^{-1}$ give you both the predictive mean and the predictive variance at any test point. The hard part is $(K + \sigma^2 I)^{-1}$, which costs $O(n^3)$ to compute and $O(n^2)$ to store.

## Choosing the kernel

The kernel encodes prior beliefs about the function:

- **RBF / squared exponential**: $k(x, x') = \sigma_f^2 \exp(-\|x - x'\|^2 / 2 \ell^2)$. Smooth functions, infinite differentiability. Default.
- **Matern**: lets you tune smoothness via a half-integer parameter $\nu$. $\nu = 5/2$ is a common modern default; smoother than $\nu = 3/2$, less restrictive than RBF.
- **Periodic**: $k(x, x') = \sigma_f^2 \exp(-2 \sin^2(\pi |x - x'| / p) / \ell^2)$. For periodic data.
- **Linear**: $k(x, x') = x^\top x'$. Recovers Bayesian linear regression.
- **Sums and products** of valid kernels are valid kernels. Add a periodic and an RBF for "trend plus seasonality."

Hyperparameters (lengthscale $\ell$, variance $\sigma_f^2$, noise $\sigma^2$) are typically learned by maximizing the marginal likelihood:

$$
\log p(\mathbf{y} \mid X, \theta) = -\tfrac{1}{2} \mathbf{y}^\top (K + \sigma^2 I)^{-1} \mathbf{y} - \tfrac{1}{2} \log |K + \sigma^2 I| - \tfrac{n}{2} \log 2\pi.
$$

## Scaling: sparse and approximate variants

- **Inducing points** ([Snelson & Ghahramani, 2005](https://papers.nips.cc/paper_files/paper/2005/hash/4491777b1aa8b5b32c2e8666dbe1a495-Abstract.html)). Pick $m \ll n$ inducing inputs, approximate $K$ via a low-rank decomposition. $O(n m^2)$ training, $O(m^2)$ prediction.
- **SVGP** ([Hensman et al., 2013](https://arxiv.org/abs/1309.6835)). Variational inference over inducing point values. Stochastic mini-batch training. The standard for large-data GPs.
- **KISS-GP / structured kernels** ([Wilson & Nickisch, 2015](https://arxiv.org/abs/1503.01057)). Exploit grid structure for $O(n)$ inference.
- **Deep kernels**. Replace $k(x, x')$ with $k(\phi(x), \phi(x'))$ where $\phi$ is a neural network. Combines deep features with kernel-conditional uncertainty.

## Where GPs are still the right tool

- **Bayesian optimization** of expensive functions (hyperparameter search, materials discovery, drug design). The acquisition function uses the posterior variance to balance exploration and exploitation.
- **Geospatial / time-series modeling** with structured covariance (kriging is just a GP).
- **Small-data regression** where calibrated uncertainty matters more than predictive accuracy.
- **Probabilistic numerics** (treat numerical algorithms as Bayesian inference).

For most large-scale supervised learning, deep nets with bootstrapping or deep ensembles deliver competitive uncertainty estimates at much better scaling.

## Common pitfalls

- **Using RBF without setting the lengthscale**. Default lengthscale produces nearly-flat predictions or wildly oscillating ones. Always optimize.
- **Ignoring the noise term $\sigma^2$**. Without it, $K$ is often singular and inversion fails. Add a small jitter even for noiseless data.
- **Reading posterior variance as "model uncertainty."** GP variance is uncertainty in the function value given the kernel and the prior. Misspecified kernels give miscalibrated variance.
- **Treating GPs as automatic.** Kernel choice is a strong prior. The model can fail silently if the kernel does not match the data structure.

## Related

- [Kernel methods and the kernel trick](/concepts/kernel-methods-and-the-kernel-trick/).
- [Bayes' rule and the posterior](/concepts/bayes-rule-and-posterior/).
- [Maximum likelihood estimation](/concepts/maximum-likelihood-estimation/).
