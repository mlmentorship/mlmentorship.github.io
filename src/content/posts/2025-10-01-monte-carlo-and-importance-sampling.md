---
title: "Monte Carlo and importance sampling"
description: "Estimate expectations by averaging over random samples. The simplest way to compute integrals you can't compute analytically."
date: "2025-10-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Monte Carlo estimates an expectation by averaging sampled contributions; importance sampling changes the sampling law and corrects those contributions.
The proposal must cover the relevant target contribution, and useful uncertainty calculations require suitable moment and dependence conditions.
Normalized weights, unbiased expectations, and diagnostic effective sample size are different objects.

## Plain Monte Carlo

For iid $X_i\sim p$ and integrable f, $\hat\mu=n^{-1}\sum_i f(X_i)$ is unbiased.
If contribution variance $\sigma^2$ is finite, $Var(\hat\mu)=\sigma^2/n$ and the iid CLT supports asymptotic normal approximations.
The $n^{-1/2}$ standard-error rate does not remove dimension-dependent variance constants or guarantee superiority over deterministic integration.

Let outcomes be zero and one, target $p=(0.9,0.1)$, and $f=(0,10)$.
Then target mean is one and direct-sampling contribution variance is $0.1(100)-1=9$.

## Importance sampling

For iid proposal draws from q and known normalized p, use:

$$
\mathbb{E}_p[f(X)] = \mathbb{E}_q\!\left[ f(X) \frac{p(X)}{q(X)} \right] \approx \frac{1}{n} \sum_i f(X_i) w_i, \quad w_i = \frac{p(X_i)}{q(X_i)}.
$$

Choose $q=(0.5,0.5)$ for the fixture.
Weights become $(1.8,0.2)$ and corrected contributions $(0,2)$, each appearing with probability one-half.
Their mean remains one but variance becomes $0.5(4)-1=1$.
Draws `[0,1,0,1]` give ordinary estimate one; exact four-draw estimator variance is $1/4$, versus $9/4$ under direct sampling.
This particular proposal improves variance; other proposals can worsen it or give infinite variance.

Require q positive wherever $pf$ contributes, and integrability of the weighted contribution.
Proposal `(1,0)` misses the nonzero contribution entirely: every estimate is zero although the target mean is one.

### Lose the Target Normalizing Constant

**New task:** Evaluate only $\tilde p=3p$.

**Picture and change:** Use self-normalized estimate $\sum_i w_if_i/\sum_iw_i$ so the common factor cancels.
This requires target coverage and a positive denominator, plus appropriate convergence conditions.

**Check:** The four fixture weights are $5.4,0.6,5.4,0.6$; numerator and denominator both equal twelve.
Dividing only that numerator by four gives three, not the desired expectation one.

**Use when:** The target density is known only up to a common factor.
The ratio is generally finite-sample biased, although consistent under suitable conditions.

### Move Proposal Mass toward the Integrand

**New task:** Replace q with $(0.1,0.9)$, keeping p and f fixed.

**Picture and change:** Recompute weights as $(9,1/9)$ and corrected contributions as $(0,10/9)$.

**Check:** Contribution mean is still one and variance is $0.9(10/9)^2-1=1/9$.
Old proposal weights cannot be reused.

**Use when:** Proposal draws can concentrate where the integrand contributes; validate coverage and weighted moments rather than weights alone.

## Effective sample size

A diagnostic of nonnegative-weight concentration is:

$$
\mathrm{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}.
$$

For illustrative weights `[1,1,1,1,6]`, normalized shares are $10\%,10\%,10\%,10\%,60\%$ and ESS is $100/40=2.5$.
These are sample weight shares, not each sample's fraction of a general estimate, which also depends on f.

<!-- visual:importance-weights-effective-sample-size -->
<figure class="learning-figure plot-panel" aria-labelledby="importance-weights-title">
	<p class="visual-kicker">Importance weights</p>
	<p class="visual-title" id="importance-weights-title">Five draws remain five draws even when unequal weights yield ESS 2.5</p><div class="visual-argument" data-visual-argument="importance-weights-effective-sample-size" data-argument-kind="ledger"><p class="visual-argument-premise">Reweighting does not create observations</p><svg viewBox="0 0 360 242" role="img" aria-label="Reweighting does not create observations. Five weights (6,1,1,1,1) sum to 10. ESS is 10 squared divided by 40, or 2.5; the actual draw count stays five."><title>Reweighting does not create observations</title><desc>Five weights (6,1,1,1,1) sum to 10. ESS is 10 squared divided by 40, or 2.5; the actual draw count stays five.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Five actual draws; illustrative</tspan><tspan x="20" dy="18">weights</tspan></text><rect data-proof-token="6" x="20" y="76" width="60" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="50" y="97" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">6</tspan></text><rect data-proof-token="1" x="84" y="76" width="60" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="114" y="97" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="114" dy="0">1</tspan></text><rect data-proof-token="1" x="148" y="76" width="60" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="178" y="97" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">1</tspan></text><rect data-proof-token="1" x="212" y="76" width="60" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="242" y="97" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="242" dy="0">1</tspan></text><rect data-proof-token="1" x="276" y="76" width="60" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="306" y="97" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="306" dy="0">1</tspan></text><text x="20" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Effective sample size</tspan></text><rect data-proof-token="100" x="20" y="174" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="50" y="195" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">100</tspan></text><rect data-proof-token="/" x="84" y="174" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="114" y="195" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="114" dy="0">/</tspan></text><rect data-proof-token="40" x="148" y="174" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="178" y="195" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">40</tspan></text><rect data-proof-token="=" x="212" y="174" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="242" y="195" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="242" dy="0">=</tspan></text><rect data-proof-token="2.5" x="276" y="174" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="306" y="195" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="306" dy="0">2.5</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
		<text class="viz-axis-label" x="12" y="206">NORMALIZE · EACH BAR'S SHARE OF TOTAL WEIGHT</text>
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> draw from q, then compute importance ratios. The fifth sample has 60% of total weight, not necessarily 60% of the final estimate. ESS is a concentration diagnostic, not a literal independent sample count. Original illustration follows <a href="https://artowen.su.domains/mc/Ch-var-is.pdf">Owen</a>, <a href="https://jmlr.org/papers/v25/19-556.html">Vehtari et al.</a>, and <a href="https://arxiv.org/abs/1809.04129">Elvira et al.</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="importance-weights-effective-sample-size"><strong>Check:</strong> Why are there five actual draws but only 2.5 effective samples?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="importance-weights-effective-sample-size">Five weights (6,1,1,1,1) sum to 10. ESS is 10 squared divided by 40, or 2.5; the actual draw count stays five.</span></figcaption>
</figure>

Equal weights give ESS n; one dominating weight pushes it toward one.
This diagnostic cannot reveal missing support, guarantee error bars, or replace weighted-integrand variance checks.
It is distinct from autocorrelation-based ESS for dependent Monte Carlo chains.

<!-- learning:core-end -->
<div data-learning-extension>

## Where it shows up in ML

| Use case | What's the integral |
|----------|---------------------|
| REINFORCE policy gradient | $\nabla_\theta \mathbb{E}_{\pi_\theta}[R]$ |
| Variational inference (ELBO gradient via reparam alternatives) | $\nabla_\phi \mathbb{E}_{q_\phi}[\log p - \log q]$ |
| Off-policy RL (importance sampling correction) | $\mathbb{E}_{\pi_\text{behavior}}[\tfrac{\pi_\text{target}}{\pi_\text{behavior}} R]$ |
| Bayesian posterior predictive | $\int p(y \mid \theta) p(\theta \mid D)\, d\theta$ |
| Importance-weighted autoencoders (IWAE) | tighter ELBO via $K$-sample IS |

## Variance reduction

- **Control variates:** subtract $c(g-E[g])$ while preserving expectation; select c and account for its estimation.
- **Antithetic variates:** preserve each sample's marginal law; benefit requires suitable negative contribution covariance.
- **Stratification:** sample strata and combine with target mass weights, not an arbitrary unweighted average.
- **Rao-Blackwellization:** conditional expectations reduce variance when their required moments exist.

## Common pitfalls

- **Heavy tails:** finite target expectation does not imply finite weighted-contribution variance.
- **Changing proposals:** reuse requires the correct proposal accounting; trajectory likelihood ratios need their specified sequential factorization.
- **False normal coverage:** $\hat\mu\pm1.96\hat\sigma/\sqrt n$ is an asymptotic iid approximation under suitable moments, not universal finite-sample coverage.
- **Finite-sample bias:** with one draw, self-normalization returns f(X), whose q-expectation need not match the p-expectation.

</div>
