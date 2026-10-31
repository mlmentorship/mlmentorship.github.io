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
	<p class="visual-title" id="gp-conditioning-title">Conditioning narrows uncertainty where observed points are informative</p><div class="visual-argument" data-visual-argument="gp-conditioning-local-uncertainty" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 439" role="img" aria-label="An exact observation removes uncertainty locally. A noiseless observation at zero locally constrains this unit RBF prior. Posterior standard deviation is sqrt(1-exp(-x squared)); uncertainty grows with distance."><title>An exact observation removes uncertainty locally</title><desc>A noiseless observation at zero locally constrains this unit RBF prior. Posterior standard deviation is sqrt(1-exp(-x squared)); uncertainty grows with distance.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">-3</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">3</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">1.1</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="0" d="M189 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><text x="189" y="245" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="189" dy="0">0</tspan></text><path data-proof-series="Prior standard deviation" d="M48 65.45454545454547L330 65.45454545454547" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="0" data-proof-y="1"><circle cx="189" cy="65.45454545454547" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Posterior standard deviation" d="M48 65.46408196091312L52.40625 65.47113646436307L56.8125 65.48290693401017L61.21875 65.50218432007435L65.625 65.53317328959241L70.03125 65.58206663178629L74.4375 65.65777738409949L78.84375 65.77283534895835L83.25 65.94443627648508L87.65625 66.19560629044531L92.0625 66.55641250539801L96.46875 67.06511650870655L100.875 67.76913561488911L105.28125 68.72565392865212L109.6875 70.00171768822975L114.09375 71.67366200487803L118.5 73.8257509587321L122.90625 76.54796835399759L127.3125 79.93296702283831L131.71875 84.07226260736414L136.125 89.05183444259256L140.53125 94.94736321924114L144.9375 101.81938584146616L149.34375 109.70867775486916L153.75 118.63217960010653L158.15625 128.5797678568483L162.5625 139.51212931928015L166.96875 151.35993933883458L171.375 164.02446760905755L175.78125 177.37964782740727L180.1875 191.27555477835455L184.59375 205.54314080344815L189 220L193.40625 205.54314080344815L197.8125 191.27555477835455L202.21875 177.37964782740727L206.625 164.02446760905755L211.03125 151.35993933883458L215.4375 139.51212931928015L219.84375 128.5797678568483L224.25 118.63217960010653L228.65625 109.70867775486916L233.0625 101.81938584146616L237.46875 94.94736321924114L241.875 89.05183444259256L246.28125 84.07226260736414L250.6875 79.93296702283831L255.09375 76.54796835399759L259.5 73.8257509587321L263.90625 71.67366200487803L268.3125 70.00171768822975L272.71875 68.72565392865212L277.125 67.76913561488911L281.53125 67.06511650870655L285.9375 66.55641250539801L290.34375 66.19560629044531L294.75 65.94443627648508L299.15625 65.77283534895835L303.5625 65.65777738409949L307.96875 65.58206663178629L312.375 65.53317328959241L316.78125 65.50218432007435L321.1875 65.48290693401017L325.59375 65.47113646436307L330 65.46408196091312" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="0" data-proof-y="0"><circle cx="189" cy="220" r="5" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">input distance from observed x=0</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">predictive standard deviation</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Prior standard deviation</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">1 at 0</tspan></text><path d="M20 368L48 368" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="372" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Posterior standard deviation</tspan></text><text x="60" y="395" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="60" dy="0">0 at 0</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">start at a query location and ask how similar it is to the observed inputs. Nearby, the covariance vector <var>k</var><sub>*</sub> is large: observations pull the mean and the subtracted correction in &Sigma;(x<sub>*</sub>) removes much of the prior uncertainty. Far away, <var>k</var><sub>*</sub> approaches zero, so the mean relaxes toward the prior and the uncertainty band widens toward the prior variance. This behavior is conditional on the chosen kernel; it is not a universal guarantee of calibration. Original schematic checked against <a href="https://gaussianprocess.org/gpml/">Rasmussen and Williams (2006)</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="gp-conditioning-local-uncertainty"><strong>Check:</strong> Which covariance term makes the posterior band narrower at the marked query? What uncertainty does that band represent?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="gp-conditioning-local-uncertainty">A noiseless observation at zero locally constrains this unit RBF prior. Posterior standard deviation is sqrt(1-exp(-x squared)); uncertainty grows with distance.</span></figcaption>
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
