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
	<p class="visual-title" id="em-bound-title">Raise a touching lower bound to prevent the likelihood from falling</p><div class="visual-argument" data-visual-argument="em-lower-bound-climb" data-argument-kind="curve"><p class="visual-argument-premise">Maximize a bound that touches the current objective</p><svg viewBox="0 0 360 375" role="img" aria-label="Maximize a bound that touches the current objective. This illustrative bound touches at 0.5. Its maximum at 0.25 raises the objective from -0.25 to -0.0625."><title>Maximize a bound that touches the current objective</title><desc>This illustrative bound touches at 0.5. Its maximum at 0.25 raises the objective from -0.25 to -0.0625.</desc><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">-1</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">1</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">-4</tspan></text><path data-proof-series="Objective: -theta squared" d="M48 92.5L52.40625 89.88525390625L56.8125 87.353515625L61.21875 84.90478515625L65.625 82.5390625L70.03125 80.25634765625L74.4375 78.056640625L78.84375 75.93994140625L83.25 73.90625L87.65625 71.95556640625L92.0625 70.087890625L96.46875 68.30322265625L100.875 66.6015625L105.28125 64.98291015625L109.6875 63.447265625L114.09375 61.99462890625L118.5 60.625L122.90625 59.33837890625L127.3125 58.134765625L131.71875 57.01416015625L136.125 55.9765625L140.53125 55.02197265625L144.9375 54.150390625L149.34375 53.36181640625L153.75 52.65625L158.15625 52.03369140625L162.5625 51.494140625L166.96875 51.03759765625L171.375 50.6640625L175.78125 50.37353515625L180.1875 50.166015625L184.59375 50.04150390625L189 50L193.40625 50.04150390625L197.8125 50.166015625L202.21875 50.37353515625L206.625 50.6640625L211.03125 51.03759765625L215.4375 51.494140625L219.84375 52.03369140625L224.25 52.65625L228.65625 53.36181640625L233.0625 54.150390625L237.46875 55.02197265625L241.875 55.9765625L246.28125 57.01416015625L250.6875 58.134765625L255.09375 59.33837890625L259.5 60.625L263.90625 61.99462890625L268.3125 63.447265625L272.71875 64.98291015625L277.125 66.6015625L281.53125 68.30322265625L285.9375 70.087890625L290.34375 71.95556640625L294.75 73.90625L299.15625 75.93994140625L303.5625 78.056640625L307.96875 80.25634765625L312.375 82.5390625L316.78125 84.90478515625L321.1875 87.353515625L325.59375 89.88525390625L330 92.5" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></path><path data-proof-series="Touching bound at theta=.5" d="M48 188.125L52.40625 181.5673828125L56.8125 175.17578125L61.21875 168.9501953125L65.625 162.890625L70.03125 156.9970703125L74.4375 151.26953125L78.84375 145.7080078125L83.25 140.3125L87.65625 135.0830078125L92.0625 130.01953125L96.46875 125.1220703125L100.875 120.390625L105.28125 115.8251953125L109.6875 111.42578125L114.09375 107.1923828125L118.5 103.125L122.90625 99.2236328125L127.3125 95.48828125L131.71875 91.9189453125L136.125 88.515625L140.53125 85.2783203125L144.9375 82.20703125L149.34375 79.3017578125L153.75 76.5625L158.15625 73.9892578125L162.5625 71.58203125L166.96875 69.3408203125L171.375 67.265625L175.78125 65.3564453125L180.1875 63.61328125L184.59375 62.0361328125L189 60.625L193.40625 59.3798828125L197.8125 58.30078125L202.21875 57.3876953125L206.625 56.640625L211.03125 56.0595703125L215.4375 55.64453125L219.84375 55.3955078125L224.25 55.3125L228.65625 55.3955078125L233.0625 55.64453125L237.46875 56.0595703125L241.875 56.640625L246.28125 57.3876953125L250.6875 58.30078125L255.09375 59.3798828125L259.5 60.625L263.90625 62.0361328125L268.3125 63.61328125L272.71875 65.3564453125L277.125 67.265625L281.53125 69.3408203125L285.9375 71.58203125L290.34375 73.9892578125L294.75 76.5625L299.15625 79.3017578125L303.5625 82.20703125L307.96875 85.2783203125L312.375 88.515625L316.78125 91.9189453125L321.1875 95.48828125L325.59375 99.2236328125L330 103.125" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-dasharray:6 3"></path><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">parameter; current 0.5, next</tspan><tspan x="190" dy="18">0.25</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">objective or lower bound</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Objective: -theta squared</tspan></text><path d="M20 336L48 336" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="340" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Touching bound at theta=.5</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">exact E makes the dashed bound touch observed log likelihood at the current parameter. M improves that fixed bound. Likelihood remains above it, proving nondecrease under the stated assumptions. The next exact E-step closes the new gap.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="em-lower-bound-climb"><strong>Check:</strong> Which two inequalities ensure the new likelihood cannot fall after the bound is improved?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="em-lower-bound-climb">This illustrative bound touches at 0.5. Its maximum at 0.25 raises the objective from -0.25 to -0.0625.</span></figcaption>
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
