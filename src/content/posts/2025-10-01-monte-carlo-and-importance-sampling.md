---
title: "Monte Carlo and importance sampling"
description: "Estimate expectations by averaging over random samples. The simplest way to compute integrals you can't compute analytically."
date: "2025-10-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Monte Carlo** estimates $\mathbb{E}_p[f(X)] = \int f(x) p(x)\, dx$ by drawing $X_1, \dots, X_n \sim p$ and computing the sample average $\hat\mu = \tfrac{1}{n} \sum_i f(X_i)$. **Importance sampling** corrects for sampling from a different distribution $q$ by reweighting: $\hat\mu = \tfrac{1}{n} \sum_i f(X_i) \tfrac{p(X_i)}{q(X_i)}$.

Almost every probabilistic ML algorithm computes intractable expectations: posterior expectations in Bayesian inference, gradients of expectations in REINFORCE, off-policy returns in RL. Monte Carlo and importance sampling are the universal hammers when you can sample from the relevant distribution but can't integrate analytically.

## Plain Monte Carlo

If $X_i \sim p$ are i.i.d.:

- **Unbiased**: $\mathbb{E}[\hat\mu] = \mathbb{E}_p[f(X)]$.
- **Variance**: $\mathrm{Var}(\hat\mu) = \mathrm{Var}_p(f) / n$.
- **Convergence rate**: $\sqrt{n}$ regardless of dimension. (Beats deterministic numerical integration in high dimensions.)

The CLT gives Gaussian confidence intervals: $\hat\mu \pm 1.96 \cdot \hat\sigma / \sqrt{n}$.

## Importance sampling

When sampling from $p$ is hard (e.g., posterior, rare event), sample from a proposal $q$ instead:

$$
\mathbb{E}_p[f(X)] = \mathbb{E}_q\!\left[ f(X) \frac{p(X)}{q(X)} \right] \approx \frac{1}{n} \sum_i f(X_i) w_i, \quad w_i = \frac{p(X_i)}{q(X_i)}.
$$

The weights $w_i$ are **importance weights**. Critical:

- **$q$ must be positive wherever $p \cdot f$ is non-zero** (no holes).
- The variance of the estimator depends on $\mathrm{Var}_q[(f \cdot p/q)]$. A poorly matched $q$ can make variance enormous.
- **Self-normalized IS**: when $p$ is only known up to a constant, use $\hat\mu = \sum w_i f(X_i) / \sum w_i$. Biased (consistent) but always usable.

## Effective sample size

A diagnostic for how well IS is working:

$$
\mathrm{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}.
$$

**Learning objective:** trace how equal draws from a proposal become unequal contributions after multiplying by $p(x)/q(x)$, and use effective sample size to recognize weight concentration.

<!-- visual:importance-weights-effective-sample-size -->
<figure class="learning-figure plot-panel" aria-labelledby="importance-weights-title">
	<p class="visual-kicker">Importance weights</p>
	<p class="visual-title" id="importance-weights-title">Five proposal draws can carry much less than five draws' worth of information.</p>
	<svg viewBox="0 0 360 390" role="img" aria-labelledby="importance-weights-svg-title importance-weights-svg-desc">
		<title id="importance-weights-svg-title">Five proposal samples become unequal weighted contributions</title>
		<desc id="importance-weights-svg-desc">Five samples drawn from proposal q initially count once each. Their p over q importance ratios are 1, 1, 1, 1, and 6. After normalization, the first four samples each contribute 10 percent and the fifth contributes 60 percent. The effective sample size is 10 squared divided by 40, or 2.5, despite a nominal sample count of five.</desc>
		<text class="viz-axis-label" x="12" y="18">DRAW FROM PROPOSAL q · FIVE SAMPLES, EACH COUNTED ONCE</text>
		<rect class="viz-plot-bg" x="10" y="28" width="340" height="62" rx="5"></rect>
		<circle class="viz-node viz-node--input" cx="64" cy="56" r="16"></circle>
		<circle class="viz-node viz-node--input" cx="122" cy="56" r="16"></circle>
		<circle class="viz-node viz-node--input" cx="180" cy="56" r="16"></circle>
		<circle class="viz-node viz-node--input" cx="238" cy="56" r="16"></circle>
		<circle class="viz-node viz-node--input" cx="296" cy="56" r="16"></circle>
		<text class="viz-node-label" x="64" y="61">1</text>
		<text class="viz-node-label" x="122" y="61">2</text>
		<text class="viz-node-label" x="180" y="61">3</text>
		<text class="viz-node-label" x="238" y="61">4</text>
		<text class="viz-node-label" x="296" y="61">5</text>
		<text class="viz-axis-label" x="12" y="116">MULTIPLY BY IMPORTANCE RATIO w = p(x) / q(x)</text>
		<rect class="viz-plot-bg" x="10" y="126" width="340" height="54" rx="5"></rect>
		<rect class="viz-node viz-node--focus" x="43" y="137" width="42" height="30" rx="4"></rect>
		<rect class="viz-node viz-node--focus" x="101" y="137" width="42" height="30" rx="4"></rect>
		<rect class="viz-node viz-node--focus" x="159" y="137" width="42" height="30" rx="4"></rect>
		<rect class="viz-node viz-node--focus" x="217" y="137" width="42" height="30" rx="4"></rect>
		<rect class="viz-node viz-node--focus" x="275" y="137" width="42" height="30" rx="4"></rect>
		<text class="viz-node-label" x="64" y="157">1</text>
		<text class="viz-node-label" x="122" y="157">1</text>
		<text class="viz-node-label" x="180" y="157">1</text>
		<text class="viz-node-label" x="238" y="157">1</text>
		<text class="viz-node-label" x="296" y="157">6</text>
		<text class="viz-axis-label" x="12" y="206">NORMALIZE · EACH BAR'S SHARE OF THE ESTIMATE</text>
		<rect class="viz-plot-bg" x="10" y="216" width="340" height="112" rx="5"></rect>
		<path class="viz-axis" d="M36 307H324"></path>
		<rect class="viz-node viz-node--output" x="45" y="292" width="38" height="15"></rect>
		<rect class="viz-node viz-node--output" x="103" y="292" width="38" height="15"></rect>
		<rect class="viz-node viz-node--output" x="161" y="292" width="38" height="15"></rect>
		<rect class="viz-node viz-node--output" x="219" y="292" width="38" height="15"></rect>
		<rect class="viz-node viz-node--output" x="277" y="217" width="38" height="90"></rect>
		<text class="viz-callout" x="64" y="284" text-anchor="middle">10%</text>
		<text class="viz-callout" x="122" y="284" text-anchor="middle">10%</text>
		<text class="viz-callout" x="180" y="284" text-anchor="middle">10%</text>
		<text class="viz-callout" x="238" y="284" text-anchor="middle">10%</text>
		<text class="viz-callout" x="296" y="238" text-anchor="middle">60%</text>
		<text class="viz-label" x="64" y="321" text-anchor="middle">sample 1</text>
		<text class="viz-label" x="122" y="321" text-anchor="middle">2</text>
		<text class="viz-label" x="180" y="321" text-anchor="middle">3</text>
		<text class="viz-label" x="238" y="321" text-anchor="middle">4</text>
		<text class="viz-label" x="296" y="321" text-anchor="middle">5</text>
		<rect class="viz-node viz-node--focus" x="10" y="344" width="340" height="36" rx="5"></rect>
		<text class="viz-callout" x="180" y="359" text-anchor="middle">nominal n = 5, but ESS = 10² / (1² + 1² + 1² + 1² + 6²)</text>
		<text class="viz-node-label" x="180" y="375">= 2.5 effective samples</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> draw from $q$, then let $p(x)/q(x)$ set each sample's contribution. Here sample 5 supplies 60% of the estimate, so five draws have an ESS diagnostic of only 2.5; ESS is a warning about concentration, not a literal new sample count. Definitions checked against <a href="https://artowen.su.domains/mc/Ch-var-is.pdf"><cite>Monte Carlo theory, methods and examples</cite></a>, <a href="https://jmlr.org/papers/v25/19-556.html">Vehtari et al.</a>, and <a href="https://arxiv.org/abs/1809.04129">Elvira et al.</a>; the weights and graphic are original.</figcaption>
</figure>

If most weight concentrates on a single sample, ESS $\approx 1$ even when $n = 10^4$. Check ESS before trusting an IS estimate.

## Where it shows up in ML

| Use case | What's the integral |
|----------|---------------------|
| REINFORCE policy gradient | $\nabla_\theta \mathbb{E}_{\pi_\theta}[R]$ |
| Variational inference (ELBO gradient via reparam alternatives) | $\nabla_\phi \mathbb{E}_{q_\phi}[\log p - \log q]$ |
| Off-policy RL (importance sampling correction) | $\mathbb{E}_{\pi_\text{behavior}}[\tfrac{\pi_\text{target}}{\pi_\text{behavior}} R]$ |
| Bayesian posterior predictive | $\int p(y \mid \theta) p(\theta \mid D)\, d\theta$ |
| Importance-weighted autoencoders (IWAE) | tighter ELBO via $K$-sample IS |

## Variance reduction

- **Control variates**: subtract a quantity with known mean: $f(X) - c(g(X) - \mathbb{E}[g])$.
- **Antithetic variates**: pair $X$ with $-X$ (for symmetric $p$).
- **Stratified sampling**: divide the domain into strata and sample within each.
- **Rao–Blackwellization**: condition out variables you can integrate exactly.

## Common pitfalls

- **Heavy-tailed importance weights.** Variance can be infinite even when the true expectation exists.
- **Reusing samples across $n$ proposals** (e.g., multi-step IS in off-policy RL): the weights compound, and variance explodes.
- **Confusing self-normalized IS bias with MC bias.** Plain MC is unbiased; self-normalized IS is biased but consistent.
- **Forgetting that $q$ must dominate $p \cdot f$.** A "small" hole in $q$ where $p$ has mass introduces unbounded bias.
