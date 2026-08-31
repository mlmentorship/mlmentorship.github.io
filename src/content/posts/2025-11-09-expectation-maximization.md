---
title: "Expectation-Maximization (EM)"
description: "Iterate between estimating latent variables given parameters (E-step) and updating parameters given latents (M-step). The standard tool for latent-variable MLE when the latents are unobserved."
date: "2025-11-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Expectation-Maximization** ([Dempster, Laird & Rubin, 1977](https://www.jstor.org/stable/2984875)) is an iterative algorithm for finding MLE / MAP parameter estimates in latent-variable models. Each iteration alternates:

1. **E-step**: compute the posterior $q(z) = p(z \mid x; \theta_t)$ over latents.
2. **M-step**: update $\theta_{t+1} = \arg\max_\theta \mathbb{E}_{q(z)}[\log p(x, z; \theta)]$.

EM monotonically increases the log-likelihood. Under standard regularity conditions, its limit points are stationary points, which may be local maxima or saddle points.

When you have a latent variable model and the latents are unobserved, direct MLE involves a marginalization $\log p(x; \theta) = \log \sum_z p(x, z; \theta)$ that is usually intractable. EM avoids this by alternately filling in expected values of $z$ and optimizing $\theta$ on the completed data.

EM underlies:

- Gaussian mixture model (GMM) fitting.
- Hidden Markov model (HMM) parameter estimation (Baum-Welch is EM).
- Probabilistic PCA, factor analysis, ICA.
- LDA topic models.
- Many missing-data imputation methods.

## The two steps

For a model with observed $x$, latent $z$, parameters $\theta$:

### E-step

Given current parameters $\theta_t$, compute the posterior over latents:

$$
q_t(z) = p(z \mid x; \theta_t).
$$

For models with conjugate or finite latent structure, this is closed-form (GMM: posterior responsibilities; HMM: forward-backward).

### M-step

Update $\theta$ to maximize the **expected complete-data log-likelihood**:

$$
\theta_{t+1} = \arg\max_\theta \mathbb{E}_{z \sim q_t}[\log p(x, z; \theta)].
$$

Often this expectation reduces to weighted MLE over the data with the latents replaced by their expected values.

## Why it works

EM maximizes a lower bound on $\log p(x; \theta)$:

$$
\log p(x; \theta) = \mathbb{E}_q[\log p(x, z; \theta)] - \mathbb{E}_q[\log q(z)] + \mathrm{KL}(q \| p_\theta(z \mid x)).
$$

The first two terms are the **ELBO** (evidence lower bound). The KL is non-negative. EM:

- E-step sets $q = p_\theta(z \mid x)$ → KL = 0 → ELBO = $\log p(x; \theta)$.
- M-step maximizes ELBO over $\theta$ holding $q$ fixed → never decreases $\log p(x; \theta)$.

So $\log p(x; \theta_{t+1}) \ge \log p(x; \theta_t)$. This monotonicity does not guarantee a global or even local maximum; EM can approach a saddle point.

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
	<figcaption><strong>Read it this way:</strong> the E-step chooses the posterior so the dashed bound touches the solid likelihood at <code>θ_t</code>. The M-step holds that bound fixed and moves to a parameter where it is no lower. Because the likelihood always lies above the bound, its value at <code>θ_{t+1}</code> cannot be below its value at <code>θ_t</code>. The next E-step closes the new gap.</figcaption>
</figure>

## Canonical example: GMM

A GMM has $K$ Gaussian components with mixture weights $\pi_k$, means $\mu_k$, covariances $\Sigma_k$. Latent $z_i \in \{1, \dots, K\}$ assigns each point to a component.

**E-step**: posterior responsibility of component $k$ for point $i$:

$$
r_{ik} = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}.
$$

**M-step**: weighted MLE updates:

$$
\mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \quad \pi_k = \frac{1}{n} \sum_i r_{ik}, \quad \Sigma_k = \frac{\sum_i r_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i r_{ik}}.
$$

Iterate until log-likelihood stabilizes.

## Variants

- **Hard-assignment EM** (k-means): replace soft responsibilities with hard 0/1 assignments. k-means is EM for a GMM with shared identity covariance and $\pi_k = 1/K$.
- **Stochastic EM**: sample $z_i$ instead of computing expected values; useful for large $z$.
- **Variational EM**: replace exact posterior with a variational approximation; modern incarnation is the VAE.
- **MAP-EM**: include a prior $p(\theta)$; M-step maximizes $\log p(\theta) + \mathbb{E}_q[\log p(x, z; \theta)]$.

## Limitations

- **Local optima**: EM converges to a local maximum; results depend on initialization. Multiple random restarts are standard.
- **Slow near optimum**: linear convergence; gets sluggish near a flat region.
- **Requires known model structure**: number of components, latent dimensions, etc.

## Common pitfalls

- **Initializing GMM means at the same point.** All components collapse to the same Gaussian; initialize by k-means++ or random data points.
- **Singular covariances.** A component centered on a single point gets $\Sigma_k \to 0$; log-likelihood diverges. Add regularization $\Sigma_k + \varepsilon I$.
- **Comparing log-likelihood across runs without checking convergence.** EM monotonically increases it within a run, but different inits give different local optima.
- **Confusing EM with k-means.** k-means is hard-assignment EM with restricted GMM; EM gives soft assignments and arbitrary covariances.

## Related

- [Gaussian mixture models](/concepts/gaussian-mixture-models/). The canonical EM application.
- [Hidden Markov models](/concepts/hidden-markov-models/). Sequential model trained by EM (Baum-Welch).
