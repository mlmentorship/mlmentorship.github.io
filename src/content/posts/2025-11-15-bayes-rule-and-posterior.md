---
title: "Bayes' rule and the posterior"
description: "Normalize prior-weighted likelihoods over the right evidence population, then distinguish changed base rates from genuinely new evidence."
date: "2025-11-15"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Bayes' rule conditions a declared probability model on observed evidence.
For hypothesis or parameter $\theta$, multiply its prior weight by its evidence likelihood, then normalize:

$$
p(\theta \mid D) = \frac{p(D \mid \theta)\, p(\theta)}{p(D)} \propto p(D \mid \theta)\, p(\theta).
$$

The evidence normalizer must be positive and finite.
For discrete hypotheses, sum their weighted likelihoods; continuous parameters require an integral under the declared measure.
Posterior probabilities, a posterior mode, and a chosen action are different outputs.

## The classic example

Consider a hypothetical test with sensitivity $P(+\mid disease)=0.99$ and specificity $P(-\mid healthy)=0.99$.
Assume prevalence $0.001$ and random sampling from that population; these are teaching assumptions, not measured clinical evidence.
In $100000$ people, expected branch counts are $100$ diseased and $99900$ healthy.
The branches contribute $99$ and $999$ positive results respectively, so the positive-result population has $1098$ members.

$$
P(disease\mid+)=\frac{99}{99+999}=\frac{11}{122}\approx0.090164.
$$

<!-- visual:bayes-positive-test-denominator -->
<figure class="learning-figure plot-panel" aria-labelledby="bayes-denominator-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="bayes-denominator-title">Why can 99% sensitivity and specificity still leave most positive results false?</p>
	<svg viewBox="0 0 360 412" role="img" aria-labelledby="bayes-denominator-svg-title bayes-denominator-svg-desc">
		<title id="bayes-denominator-svg-title">Natural-frequency funnel for a rare disease test</title>
		<desc id="bayes-denominator-svg-desc">Start with 100,000 people. At 0.1 percent prevalence, 100 have the disease and 99,900 are healthy. With 99 percent sensitivity, the diseased branch contributes 99 true-positive results. With 99 percent specificity, one percent of the healthy branch contributes 999 false-positive results. A proportional positive-result strip contains a narrow 99-person true-positive segment and a 999-person false-positive segment. Both enter the denominator 1,098. The true-positive share is 99 divided by 1,098, about 9 percent.</desc>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="404" rx="6"></rect>
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
		<path d="M86 254V269H38V288M274 254V269H194V288" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<rect x="24" y="290" width="28.131147541" height="24" data-count="99" style="fill:var(--viz-output-stroke)"></rect>
		<rect x="52.131147541" y="290" width="283.868852459" height="24" data-count="999" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-dasharray:4 2"></rect>
		<text x="194" y="306" style="font-family:var(--font-ui);font-size:12px;fill:var(--c-text);text-anchor:middle">999 false positives</text>
		<text x="24" y="333" style="font-family:var(--font-ui);font-size:12px;fill:var(--c-text);text-anchor:start">99 true</text>
		<path d="M24 339V345H336V339" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
		<text x="180" y="363" style="font-family:var(--font-ui);font-size:12px;fill:var(--c-text);text-anchor:middle">all 1,098 positive results</text>
		<text x="180" y="389" style="font-family:var(--font-ui);font-size:17px;font-weight:700;fill:var(--c-text);text-anchor:middle">99 / (99 + 999) = 9%</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> the narrow solid piece is 99 true positives. The much wider dashed piece is 999 false positives. Their widths use the same people-per-unit scale. The denominator is the entire strip, not just the diseased branch: 99 divided by 1,098, about 9%. These are expected counts under the hypothetical test assumptions. <span class="visual-reader-check" data-reader-check="bayes-positive-test-denominator"><strong>Check:</strong> Why is a positive result only about 9% likely to indicate disease despite 99% sensitivity?</span></figcaption>
</figure>

The healthy posterior is $999/1098$; together the two posterior masses sum to one.

## The four pieces

| Piece | Name | Role |
|-------|------|------|
| $p(\theta)$ | Prior | Weight before this observation |
| $p(D\mid\theta)$ | Likelihood | Evidence model at a candidate parameter |
| $p(\theta\mid D)$ | Posterior | Normalized conditional distribution |
| $p(D)$ | Evidence | Sum or integral of prior-weighted likelihoods |

The evidence is constant across parameters within one model, so maximizing an unnormalized posterior can locate its mode.
It cannot be discarded when reporting posterior probabilities or comparing model evidence.
A constant prior density can preserve an MLE maximizer in that parameterization, when the posterior and maximizer exist.
This is a prior assumption, not an absence of assumptions.

## Change the Requirement

### Change the Population

**New task:** Increase prevalence to one percent while keeping sensitivity and specificity unchanged.

**Picture and change:** Begin with $1000$ diseased and $99000$ healthy people.

**Check:** Both branches contribute $990$ positives; posterior disease probability becomes $1/2$.

**Use when:** Screening and referral populations have different base rates and the transported test properties remain justified.

### Observe Another Positive

**New task:** Add a second positive result, conditionally independent of the first given disease status.

**Picture and change:** Multiply each branch by its second-result likelihood.

**Check:** Original-population expected double-positive counts are $98.01$ and $9.99$, giving $98.01/108=0.9075$.

**Use when:** The observation supplies genuinely new conditional evidence.
Copying the same test report is perfectly dependent evidence and leaves the original posterior unchanged.

<!-- learning:core-end -->
<div data-learning-extension>

## Conjugate priors

A conjugate prior produces a posterior in the same family, with parameters updated from the observations.

| Likelihood | Conjugate prior | Posterior family |
|-----------|-----------------|------------------|
| Bernoulli/binomial | Beta | Beta |
| Categorical/multinomial | Dirichlet | Dirichlet |
| Gaussian (mean, known $\sigma^2$) | Gaussian | Gaussian |
| Gaussian precision, known mean | Gamma | Gamma |
| Poisson | Gamma | Gamma |

Used in: Thompson sampling for bandits (Beta-Bernoulli), online recsys updates, conjugate Gibbs samplers.

## Approximate inference (when conjugacy fails)

When closed-form conditioning is unavailable, distinguish the approximation from the desired posterior:

- **Laplace:** approximate a suitable interior mode with a Gaussian using local curvature.
- **Variational inference:** optimize within a chosen distribution family; the family and divergence affect the approximation.
- **MCMC:** construct a target-invariant chain; convergence and Monte Carlo uncertainty still require diagnosis.
- **Stochastic-gradient samplers:** introduce data subsampling and step-size/approximation considerations.

## Common pitfalls

- **Reversed conditioning.** A likelihood does not need to normalize over parameters.
- **Prior sensitivity.** Weak evidence can leave substantial sensitivity to the prior; small sample size alone does not prove prior dominance.
- **Mode-only reporting.** A mode omits uncertainty and can change under reparameterization.
- **Improper priors.** An unnormalized prior can produce either a proper or improper posterior; check the integral.

## Reconstruct

<details>
<summary>Why cannot two copies of one positive report be treated as independent tests?</summary>

Once the first report is known, its exact copy is certain under either disease status.
The copy supplies likelihood ratio one, not another factor of ninety-nine.

</details>

</div>
