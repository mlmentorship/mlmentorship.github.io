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
	<p class="visual-title" id="em-bound-title">Raise a touching lower bound to prevent the likelihood from falling</p><div class="visual-argument" data-visual-argument="em-lower-bound-climb" data-argument-kind="curve" data-argument-composition="bound-climb"><svg viewBox="0 0 360 511" role="img" aria-label="Maximize a bound that touches the current objective. This illustrative bound touches at 0.5. Its maximum at 0.25 raises the objective from -0.25 to -0.0625."><title>Maximize a bound that touches the current objective</title><desc>This illustrative bound touches at 0.5. Its maximum at 0.25 raises the objective from -0.25 to -0.0625.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Climb a supporting bound that touches</tspan><tspan x="20" dy="18">now</tspan></text><path d="M35 145L39.53125 141.30859375L44.0625 137.734375L48.59375 134.27734375L53.125 130.9375L57.65625 127.71484375L62.1875 124.609375L66.71875 121.62109375L71.25 118.75L75.78125 115.99609375L80.3125 113.359375L84.84375 110.83984375L89.375 108.4375L93.90625 106.15234375L98.4375 103.984375L102.96875 101.93359375L107.5 100L112.03125 98.18359375L116.5625 96.484375L121.09375 94.90234375L125.625 93.4375L130.15625 92.08984375L134.6875 90.859375L139.21875 89.74609375L143.75 88.75L148.28125 87.87109375L152.8125 87.109375L157.34375 86.46484375L161.875 85.9375L166.40625 85.52734375L170.9375 85.234375L175.46875 85.05859375L180 85L184.53125 85.05859375L189.0625 85.234375L193.59375 85.52734375L198.125 85.9375L202.65625 86.46484375L207.1875 87.109375L211.71875 87.87109375L216.25 88.75L220.78125 89.74609375L225.3125 90.859375L229.84375 92.08984375L234.375 93.4375L238.90625 94.90234375L243.4375 96.484375L247.96875 98.18359375L252.5 100L257.03125 101.93359375L261.5625 103.984375L266.09375 106.15234375L270.625 108.4375L275.15625 110.83984375L279.6875 113.359375L284.21875 115.99609375L288.75 118.75L293.28125 121.62109375L297.8125 124.609375L302.34375 127.71484375L306.875 130.9375L311.40625 134.27734375L315.9375 137.734375L320.46875 141.30859375L325 145" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2.5"></path><path d="M35 280L39.53125 270.7421875L44.0625 261.71875L48.59375 252.9296875L53.125 244.375L57.65625 236.0546875L62.1875 227.96875L66.71875 220.1171875L71.25 212.5L75.78125 205.1171875L80.3125 197.96875L84.84375 191.0546875L89.375 184.375L93.90625 177.9296875L98.4375 171.71875L102.96875 165.7421875L107.5 160L112.03125 154.4921875L116.5625 149.21875L121.09375 144.1796875L125.625 139.375L130.15625 134.8046875L134.6875 130.46875L139.21875 126.3671875L143.75 122.5L148.28125 118.8671875L152.8125 115.46875L157.34375 112.3046875L161.875 109.375L166.40625 106.6796875L170.9375 104.21875L175.46875 101.9921875L180 100L184.53125 98.2421875L189.0625 96.71875L193.59375 95.4296875L198.125 94.375L202.65625 93.5546875L207.1875 92.96875L211.71875 92.6171875L216.25 92.5L220.78125 92.6171875L225.3125 92.96875L229.84375 93.5546875L234.375 94.375L238.90625 95.4296875L243.4375 96.71875L247.96875 98.2421875L252.5 100L257.03125 101.9921875L261.5625 104.21875L266.09375 106.6796875L270.625 109.375L275.15625 112.3046875L279.6875 115.46875L284.21875 118.8671875L288.75 122.5L293.28125 126.3671875L297.8125 130.46875L302.34375 134.8046875L306.875 139.375L311.40625 144.1796875L315.9375 149.21875L320.46875 154.4921875L325 160" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5"></path><circle cx="252.5" cy="100" r="7" style="fill:var(--viz-focus-stroke);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><path d="M252.5 115L216.25 107.5M224.89450577220063 105.2038031542592L216.25 107.5L223.27366093991301 113.03788651031603" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-linejoin:round"></path><path d="M216.25 92.5L216.25 88.75" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="216.25" cy="88.75" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="20" y="395" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Contact at 0.5; bound maximum at 0.25</tspan></text><text x="20" y="458" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Objective rises: -0.25 to -0.0625</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">exact E makes the dashed bound touch observed log likelihood at the current parameter. M improves that fixed bound. Likelihood remains above it, proving nondecrease under the stated assumptions. The next exact E-step closes the new gap.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="em-lower-bound-climb"><strong>Check:</strong> Which two inequalities ensure the new likelihood cannot fall after the bound is improved?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="em-lower-bound-climb">This illustrative bound touches at 0.5. Its maximum at 0.25 raises the objective from -0.25 to -0.0625. <br><strong>Why is the true objective at the new bound maximum no lower than the old objective?</strong></span></figcaption>
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
