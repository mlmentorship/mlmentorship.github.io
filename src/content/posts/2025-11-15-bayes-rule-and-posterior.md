---
title: "Bayes' rule and the posterior"
description: "How to update beliefs given evidence: posterior ∝ likelihood × prior. The foundation of Bayesian inference, naive Bayes, and probabilistic graphical models."
date: "2025-11-15"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

For random variables $\theta$ (parameters / hypothesis) and $D$ (data / evidence):

$$
p(\theta \mid D) = \frac{p(D \mid \theta)\, p(\theta)}{p(D)} \propto p(D \mid \theta)\, p(\theta).
$$

The **posterior** $p(\theta \mid D)$ is proportional to the **likelihood** $p(D \mid \theta)$ times the **prior** $p(\theta)$.

Bayes' rule is the only mathematically consistent way to update probabilistic beliefs given new evidence. It underlies probabilistic ML (Gaussian processes, Bayesian deep learning), classification (naive Bayes), generative models (latent variable inference), and many engineering systems (Kalman filtering, sensor fusion).

The connection to MLE: the posterior peak (MAP estimate) collapses to MLE under a uniform prior. So MLE is a special case of Bayesian inference with no prior beliefs.

## The four pieces

| Piece | Name | What it is |
|-------|------|-----------|
| $p(\theta)$ | Prior | Beliefs about $\theta$ before seeing data |
| $p(D \mid \theta)$ | Likelihood | How probable the data is under each hypothesis |
| $p(\theta \mid D)$ | Posterior | Updated beliefs after seeing data |
| $p(D)$ | Evidence / marginal likelihood | Normalizing constant; $\int p(D \mid \theta) p(\theta)\, d\theta$ |

The evidence is often intractable (high-dimensional integral). For point estimates and many decisions you can ignore it.

## The classic example

A medical test is 99% accurate: $p(\text{pos} \mid \text{disease}) = 0.99$ and $p(\text{neg} \mid \text{healthy}) = 0.99$. The disease has prevalence $p(\text{disease}) = 0.001$. A random person tests positive. What is $p(\text{disease} \mid \text{pos})$?

$$
p(\text{disease} \mid \text{pos}) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} \approx 0.09.
$$

<!-- visual:bayes-positive-test-denominator -->
<figure class="learning-figure plot-panel" aria-labelledby="bayes-denominator-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="bayes-denominator-title">Why can a 99%-accurate test still make most positive results false?</p>
	<svg viewBox="0 0 360 350" role="img" aria-labelledby="bayes-denominator-svg-title bayes-denominator-svg-desc">
		<title id="bayes-denominator-svg-title">Natural-frequency funnel for a rare disease test</title>
		<desc id="bayes-denominator-svg-desc">Start with 100,000 people. At 0.1 percent prevalence, 100 have the disease and 99,900 are healthy. With 99 percent sensitivity, the diseased branch contributes 99 true-positive results. With 99 percent specificity, one percent of the healthy branch contributes 999 false-positive results. The branches merge into 1,098 positive results, of which only 99, or about 9 percent, indicate disease.</desc>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="342" rx="6"></rect>
		<rect class="viz-node viz-node--input" x="95" y="18" width="170" height="52" rx="5"></rect>
		<text class="viz-node-label" x="180" y="41">100,000 people</text>
		<text class="viz-node-value" x="180" y="58">before testing</text>
		<path d="M180 70V84M180 84H86V96M180 84H274V96" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M80 92L86 100L92 92ZM268 92L274 100L280 92Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--focus" x="12" y="100" width="148" height="62" rx="5"></rect>
		<text class="viz-node-label" x="86" y="124">100 have disease</text>
		<text class="viz-node-value" x="86" y="143">prior: 0.1%</text>
		<rect class="viz-node" x="200" y="100" width="148" height="62" rx="5" style="stroke-dasharray:5 3"></rect>
		<text class="viz-node-label" x="274" y="124">99,900 healthy</text>
		<text class="viz-node-value" x="274" y="143">prior: 99.9%</text>
		<path d="M86 162V188M274 162V188" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M80 184L86 192L92 184ZM268 184L274 192L280 184Z" style="fill:var(--viz-edge)"></path>
		<text class="viz-label" x="86" y="180" text-anchor="middle">99% test positive</text>
		<text class="viz-label" x="274" y="180" text-anchor="middle">1% test positive</text>
		<rect class="viz-node viz-node--output" x="12" y="192" width="148" height="62" rx="5"></rect>
		<text class="viz-node-label" x="86" y="216">99 true positives</text>
		<text class="viz-node-value" x="86" y="235">99% of 100</text>
		<rect class="viz-node viz-node--focus" x="200" y="192" width="148" height="62" rx="5" style="stroke-dasharray:5 3"></rect>
		<text class="viz-node-label" x="274" y="216">999 false positives</text>
		<text class="viz-node-value" x="274" y="235">1% of 99,900</text>
		<path d="M86 254V268H180V282M274 254V268H180" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M174 278L180 286L186 278Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--output" x="72" y="286" width="216" height="47" rx="23"></rect>
		<text class="viz-node-label" x="180" y="307">99 / (99 + 999) = 9%</text>
		<text class="viz-node-value" x="180" y="324">disease given a positive result</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> the rare prior creates only 100 opportunities for a true positive but 99,900 opportunities for a false positive. Both branches feed the positive-result denominator, so 999 false positives outnumber 99 true positives and only about 9% of positive results indicate disease.</figcaption>
</figure>

Despite 99% test accuracy, about 91% of positive results are false. The prior strongly affects the posterior.

## Conjugate priors

A prior is **conjugate** to a likelihood if the resulting posterior is in the same family (so updating stays in closed form).

| Likelihood | Conjugate prior | Posterior family |
|-----------|-----------------|------------------|
| Bernoulli/binomial | Beta | Beta |
| Categorical/multinomial | Dirichlet | Dirichlet |
| Gaussian (mean, known $\sigma^2$) | Gaussian | Gaussian |
| Gaussian (precision) | Gamma | Gamma |
| Poisson | Gamma | Gamma |

Used in: Thompson sampling for bandits (Beta-Bernoulli), online recsys updates, conjugate Gibbs samplers.

## Approximate inference (when conjugacy fails)

Modern Bayesian deep learning rarely has closed-form posteriors. Standard approximations:

- **Laplace approximation**: Gaussian centered at MAP with covariance from the Hessian.
- **Variational inference**: optimize a parametric family $q_\phi(\theta)$ to minimize $\mathrm{KL}(q_\phi \| p(\theta \mid D))$.
- **MCMC** (Metropolis-Hastings, HMC, NUTS): draw samples from the posterior asymptotically.
- **Stochastic-gradient Langevin / SGHMC**: scale to large data via mini-batches.

## Common pitfalls

- **Confusing likelihood with posterior.** $p(D \mid \theta)$ is not a probability distribution over $\theta$; it does not integrate to 1 over $\theta$.
- **Ignoring the prior in low-data regimes.** With small $n$, the posterior is dominated by the prior.
- **Reporting MAP without uncertainty.** A posterior contains more than its mode; the spread is often the more useful information.
- **Improper priors.** Some "uniform" priors over unbounded parameter spaces don't integrate; the posterior may still be proper (or may not be).
