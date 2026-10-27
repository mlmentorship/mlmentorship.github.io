---
title: "Expectation-Maximization (EM)"
description: "Compute posterior expected complete-data statistics, then improve a touching likelihood bound. Trace one exact two-coin EM iteration."
date: "2025-11-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

[Expectation-Maximization](https://www.jstor.org/stable/2984875) alternates posterior computation with parameter improvement for a latent-variable model.
An exact posterior makes a likelihood lower bound touch the current objective; improving that fixed bound cannot decrease observed likelihood.
This guarantee is not a global-optimum or component-identification guarantee.

## The two steps

Let x be observations, z latent variables, and theta parameters.
Require available exact posteriors, common support, finite required expectations, and an M-step that does not reduce its target.

### E-step

Given current parameters $\theta_t$, compute the posterior over latents:

$$
q_t(z) = p(z \mid x; \theta_t).
$$

Current theta stays fixed while q is computed.
Calculate expected complete-data sufficient statistics, not merely a substituted mean latent value.
For example, $E[zz^\top]$ generally differs from $E[z]E[z]^\top$.

### M-step

Update $\theta$ to maximize the **expected complete-data log-likelihood**:

$$
\theta_{t+1} = \arg\max_\theta \mathbb{E}_{z \sim q_t}[\log p(x, z; \theta)].
$$

Here q stays fixed while theta changes.
Write $Q(\theta\mid\theta_t)=E_{q_t}\log p(x,z;\theta)$.

For a teaching mixture, choose a coin independently for each two-toss group.
Initial weights are $(0.5,0.5)$ and head probabilities $(0.75,0.25)$; observe head counts two and one.
The binomial probability includes $\binom{2}{h}p^h(1-p)^{2-h}$.
Count two has likelihoods $(0.5625,0.0625)$, giving responsibilities $(0.9,0.1)$.
Count one has likelihoods $(0.375,0.375)$, giving responsibilities $(0.5,0.5)$.

Effective group counts are $(1.4,0.6)$; weighted heads are $(2.3,0.7)$.
The M-step sets weights to $(0.7,0.3)$ and head probabilities to $(2.3/2.8,0.7/1.2)=(23/28,7/12)$.
The denominators count expected tosses, not expected groups.
Observed likelihood rises from $0.3125\times0.375=0.1171875$ to approximately $0.20172548$.
Its log rises from $-2.143980$ to $-1.600848$.

## Why it works

EM maximizes a lower bound on $\log p(x; \theta)$:

$$
\log p(x; \theta) = \mathbb{E}_q[\log p(x, z; \theta)] - \mathbb{E}_q[\log q(z)] + \mathrm{KL}(q \| p_\theta(z \mid x)).
$$

Write $F(q,\theta)=Q(\theta\mid\theta_t)+H(q)$ when q is the current posterior.
The identity says $\ell(\theta)=F(q,\theta)+\mathrm{KL}(q\|p_\theta)$, with a nonnegative KL gap.
Exact E sets the gap to zero at theta_t.
Improving Q with q fixed then yields:

$$
\ell(\theta_{t+1})\ge F(q_t,\theta_{t+1})\ge F(q_t,\theta_t)=\ell(\theta_t).
$$

<!-- visual:em-lower-bound-climb -->
<figure class="learning-figure plot-panel" aria-labelledby="em-bound-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="em-bound-title">Why can an EM iteration not lower the observed-data likelihood?</p>
	<svg viewBox="0 0 360 420" role="img" aria-labelledby="em-bound-svg-title em-bound-svg-desc">
		<title id="em-bound-svg-title">EM climbs a lower bound on the observed-data log-likelihood</title>
		<desc id="em-bound-svg-desc">A solid curve is the observed-data log-likelihood. A dashed lower-bound curve touches it at the current parameter theta t because the E-step sets q t to the exact posterior and makes the KL gap zero. Holding q t fixed, the M-step moves right to theta t plus 1, where the lower bound is higher. The solid likelihood at the new parameter is at least as high as that bound. The next E-step closes the remaining gap.</desc>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="218" rx="6"></rect>
		<path class="viz-axis" d="M40 190H338M40 190V24"></path>
		<text class="viz-axis-label" x="338" y="207" text-anchor="end">parameter θ</text>
		<text class="viz-axis-label" x="45" y="20">objective</text>
		<path d="M42 170Q74 148 106 132Q154 112 201 78Q259 37 332 76" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round"></path>
		<text class="viz-callout" x="326" y="64" text-anchor="end">log p(x; θ)</text>
		<text class="viz-label" x="326" y="79" text-anchor="end">observed likelihood · solid</text>
		<path d="M42 184Q76 151 106 132Q167 99 226 104Q283 109 332 146" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-dasharray:7 5;stroke-linecap:round"></path>
		<text class="viz-callout" x="326" y="131" text-anchor="end">F(q<tspan baseline-shift="sub" font-size="8">t</tspan>, θ)</text>
		<text class="viz-label" x="326" y="146" text-anchor="end">fixed lower bound · dashed</text>
		<path d="M106 132V190M226 104V190" style="fill:none;stroke:var(--c-text-soft);stroke-width:1.2;stroke-dasharray:3 3"></path>
		<circle cx="106" cy="132" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2.5"></circle>
		<circle cx="226" cy="104" r="5" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></circle>
		<text class="viz-axis-label" x="106" y="207" text-anchor="middle">θ<tspan baseline-shift="sub" font-size="8">t</tspan></text>
		<text class="viz-axis-label" x="226" y="207" text-anchor="middle">θ<tspan baseline-shift="sub" font-size="8">t+1</tspan></text>
		<path d="M116 119Q164 91 216 99" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
		<path d="M210 93L220 100L209 104Z" style="fill:var(--viz-edge)"></path>
		<text class="viz-callout" x="158" y="157" text-anchor="middle">M-step raises bound →</text>
		<rect class="viz-node viz-node--input" x="12" y="240" width="336" height="48" rx="5"></rect>
		<text class="viz-node-label" x="180" y="259">1 · E-step: make the bound tight</text>
		<text class="viz-node-value" x="180" y="277">q<tspan baseline-shift="sub" font-size="8">t</tspan> = p(z|x; θ<tspan baseline-shift="sub" font-size="8">t</tspan>) ⇒ KL gap = 0</text>
		<path d="M180 288V302" style="stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M174 298L180 306L186 298Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--focus" x="12" y="306" width="336" height="48" rx="5" style="stroke-dasharray:7 4"></rect>
		<text class="viz-node-label" x="180" y="325">2 · M-step: improve the fixed bound</text>
		<text class="viz-node-value" x="180" y="343">F(q<tspan baseline-shift="sub" font-size="8">t</tspan>, θ<tspan baseline-shift="sub" font-size="8">t+1</tspan>) ≥ F(q<tspan baseline-shift="sub" font-size="8">t</tspan>, θ<tspan baseline-shift="sub" font-size="8">t</tspan>) = log p(x; θ<tspan baseline-shift="sub" font-size="8">t</tspan>)</text>
		<path d="M180 354V368" style="stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M174 364L180 372L186 364Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--output" x="12" y="372" width="336" height="40" rx="20"></rect>
		<text class="viz-node-label" x="180" y="389">3 · Likelihood stays above its bound</text>
		<text class="viz-node-value" x="180" y="404">log p(x; θ<tspan baseline-shift="sub" font-size="8">t+1</tspan>) ≥ F(q<tspan baseline-shift="sub" font-size="8">t</tspan>, θ<tspan baseline-shift="sub" font-size="8">t+1</tspan>)</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> exact E makes the dashed bound touch observed log likelihood at the current parameter. M improves that fixed bound. Likelihood remains above it, proving nondecrease under the stated assumptions. The next exact E-step closes the new gap.</figcaption>
</figure>

### Add a Parameter Prior

**New task:** Prefer interior coin probabilities with independent Beta(2,2) priors; leave mixture weights unregularized.

**Picture and change:** Keep posterior responsibilities, but maximize Q plus log prior.

**Check:** Probability updates become $3.3/4.8=0.6875$ and $1.7/3.2=0.53125$.

**Use when:** A MAP estimator is required.
The protected objective is log posterior, not bare observed likelihood.

### Improve Only Part of M

**New task:** Update mixture weights while keeping both coin probabilities at their initial values.

**Picture and change:** This generalized M-step improves Q without jointly maximizing every parameter.

**Check:** Weights $(0.7,0.3)$ give observed likelihood $0.4125\times0.375=0.1546875$, above the initial value.

**Use when:** A verified improving parameter block is available but joint optimization is expensive.

Identical components can remain identical; nondecreasing likelihood does not distinguish them or identify a global solution.
Under appropriate regularity and convergence conditions, limit points can be stationary, including saddles.
Unconstrained Gaussian mixtures can instead diverge through covariance collapse.

<!-- learning:core-end -->
<div data-learning-extension>

## Canonical example: GMM

[Gaussian mixtures](/concepts/gaussian-mixture-models/) apply this owner to posterior component responsibilities and weighted Gaussian sufficient statistics.
Detailed covariance updates belong to that application, not a second generic EM derivation.

## Variants

- **Hard assignment:** k-means relates to restricted mixtures through hard-assignment or small-variance limits, not ordinary finite-variance exact EM.
- **Stochastic approximations:** sampling noise need not preserve likelihood nondecrease at every iteration.
- **Variational E:** improving a restricted lower bound does not automatically increase observed likelihood; the touching-bound step may be missing.
- **MAP-EM:** an explicit prior changes the optimized objective.

## Limitations

Initialization, latent-model specification, missing information, and numerical optimization affect convergence and usefulness.
Comparing restarts does not turn a nonconvex model into a globally solved problem.

## Common pitfalls

Ad hoc covariance stabilization and approximate E/M updates require their own objective/monotonicity analysis.
Alternating two convenient steps is not enough to establish the exact EM proof.

## Related

- [Gaussian mixture models](/concepts/gaussian-mixture-models/). The canonical EM application.
- [Hidden Markov models](/concepts/hidden-markov-models/). Sequential model trained by EM (Baum-Welch).

</div>
