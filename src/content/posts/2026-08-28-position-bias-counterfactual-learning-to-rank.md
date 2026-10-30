---
title: "Position bias and counterfactual learning to rank"
description: "Clicks reflect relevance and exposure. Use randomized data, propensities, IPS, self-normalization, or doubly robust estimates without hiding support and variance limits."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["position bias", "counterfactual LTR", "unbiased learning to rank", "IPS ranking", "SNIPS", "doubly robust ranking"]
roles: ["Ranking MLE", "Applied Scientist", "Product ML"]
rounds: ["ML system design", "Evaluation", "Experimentation"]
difficulty: "Advanced"
priority: "Role-specific"
prerequisites: ["causal-inference-for-ml-decisions", "ranking-metrics-ndcg-map-mrr"]
---

## Summary

Clicks reflect both relevance and whether users examined the result under the displayed ranking.
Identical relevance can produce different click counts at different positions.
Define the desired outcome and the probability creating its observation before choosing a correction.

Examination weighting estimates a signal under a click model; policy ratios target an alternative policy's reward.
These are different estimands and different propensities.

## The observation process

A simple examination model writes click probability as

$$
P(C=1\mid q,d,k)
=
P(E=1\mid k)P(R=1\mid q,d),
$$

where $E$ means the user examined position $k$ and $R$ means document $d$ was relevant to query $q$.

Here $C,E,R$ denote click, examination, and relevance; $q,d,k$ denote query, document, and position.
The factorization excludes effects such as trust-driven clicks and examination dependencies on snippets or neighboring results.
No click does not distinguish an unexamined result from an examined rejection.

<!-- visual:position-bias-inverse-examination-weight -->
<figure class="learning-figure" aria-labelledby="position-bias-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="position-bias-visual-title">Importance weights change estimated contribution, not the observed click count</p><div class="visual-argument" data-visual-argument="position-bias-inverse-examination-weight" data-argument-kind="ledger"><p class="visual-argument-premise">The weight changes estimated contribution, not the click</p><svg viewBox="0 0 360 224" role="img" aria-label="The weight changes estimated contribution, not the click. One click under examination probability 0.2 contributes weight 5 under this estimator. It remains one observed click."><title>The weight changes estimated contribution, not the click</title><desc>One click under examination probability 0.2 contributes weight 5 under this estimator. It remains one observed click.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Observed click</tspan></text><rect data-proof-token="1" x="20" y="58" width="316" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="178" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">1</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Inverse examination contribution</tspan></text><rect data-proof-token="1" x="20" y="156" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="50" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">1</tspan></text><rect data-proof-token="/" x="84" y="156" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="114" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="114" dy="0">/</tspan></text><rect data-proof-token=".2" x="148" y="156" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="178" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">.2</tspan></text><rect data-proof-token="=" x="212" y="156" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="242" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="242" dy="0">=</tspan></text><rect data-proof-token="5" x="276" y="156" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="306" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="306" dy="0">5</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="The same illustrative relevance rate shown at a fully examined first position and a less frequently examined fourth position">
		<section class="visual-panel">
			<h4>Position 1 · usually examined</h4>
			<p>100 impressions with the same assumed 60% relevance rate</p>
			<table class="cm-grid" aria-label="Expected clicks and inverse-weighted signal at position one">
				<tbody>
					<tr><th scope="row">Examination propensity</th><td><strong>e = 1.00</strong></td></tr>
					<tr><th scope="row">Expected clicks</th><td>100 × 0.60 × 1.00 = <strong>60</strong></td></tr>
					<tr><th scope="row">Inverse weight</th><td>1 / 1.00 = <strong>1×</strong></td></tr>
					<tr><th scope="row">Weighted click signal</th><td class="cm-selected">60 × 1 = <strong>60</strong></td></tr>
				</tbody>
			</table>
		</section>
		<section class="visual-panel">
			<h4>Position 4 · often skipped</h4>
			<p>100 impressions with the same assumed 60% relevance rate</p>
			<table class="cm-grid" aria-label="Expected clicks and inverse-weighted signal at position four">
				<tbody>
					<tr><th scope="row">Examination propensity</th><td><strong>e = 0.25</strong></td></tr>
					<tr><th scope="row">Expected clicks</th><td>100 × 0.60 × 0.25 = <strong>15</strong></td></tr>
					<tr><th scope="row">Inverse weight</th><td>1 / 0.25 = <strong>4×</strong></td></tr>
					<tr><th scope="row">Weighted click signal</th><td class="cm-selected">15 × 4 = <strong>60</strong></td></tr>
				</tbody>
			</table>
		</section>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> hold relevance probability at 0.60 over 100 impressions. Examination rates 1 and 0.25 produce 60 and 15 expected clicks. Weights 1 and 4 restore expected signals of 60 under this model. These are hypothetical expectations, not measured clicks or a target-policy value estimate. Original example checked against <a href="https://arxiv.org/abs/1608.04468">Joachims, Swaminathan, and Schnabel</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="position-bias-inverse-examination-weight"><strong>Check:</strong> Why do fifteen lower-position clicks contribute sixty units in expectation without becoming sixty actual clicks?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="position-bias-inverse-examination-weight">One click under examination probability 0.2 contributes weight 5 under this estimator. It remains one observed click.</span></figcaption>
</figure>

## Estimate propensities

Examination propensity is the probability of examination; policy propensity is the probability of assigning an action or slate.
Randomized swaps or a validated click model can inform examination probabilities.
Randomized assignment should log its actual conditional policy probability, candidates, and policy version.
Propensities inferred from click outcomes alone need an identification model.

## Worked example

Fix one context and deterministic toy rewards, with a five-event log containing four A actions and one B.

| Action | Logger $\mu$ | Target $\pi$ | Reward | Policy ratio |
| --- | --- | --- | --- | --- |
| A | 0.8 | 0.2 | 1 | 0.25 |
| B | 0.2 | 0.8 | 0 | 4 |
| B under a different, unsupported logger | 0 | 0.8 | Unknown | Undefined |

The last row is a separate failure scenario, not part of the five-event log.
Raw reward is $4/5=0.8$, but the original target policy's value is $0.2(1)+0.8(0)=0.2$.

## Inverse propensity scoring

For contextual bandit data $(x_i,a_i,r_i)$ logged under policy $\mu$, the value of target policy $\pi$ can be estimated by

$$
\widehat{V}_{IPS}(\pi)
=
\frac{1}{n}\sum_{i=1}^n
\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}r_i.
$$

Here $x_i,a_i,r_i$ are context, chosen action, and mature reward; $\mu,\pi$ are logger and target policies.
The fixture gives $(4(0.25)(1)+4(0))/5=0.2$.
The action counts match logger expectations by construction, not by a precision guarantee.

With correct assignment probabilities, overlap, consistent outcomes, and no hidden assignment confounding, the expectation becomes target-policy reward.
Very small logger probabilities can create large variance; reward maturity and interference remain separate assumptions.

## Self-normalization and clipping

Self-normalized IPS, often called SNIPS, divides by the sum of weights:

$$
\widehat{V}_{SNIPS}(\pi)
=
\frac{\sum_i w_i r_i}{\sum_i w_i},
\qquad
w_i=\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}.
$$

The fixture's weight sum is $4(0.25)+4=5$, so SNIPS also equals $0.2$.
Its effective sample-size diagnostic is $5^2/(4(0.25)^2+4^2)=20/13\approx1.538$.
Self-normalization has finite-sample bias and is undefined when the realized weight sum is zero.

Weight clipping applies

$$
w_i^{\mathrm{clip}}=\min(w_i,c).
$$

Clipping changes weights, whereas SNIPS changes normalization.
Either can trade variance against bias; neither is an identification repair.

## Doubly robust estimation

Let $\hat{r}(x,a)$ predict the expected reward. A doubly robust value estimate is

$$
\widehat{V}_{DR}(\pi)
=
\frac{1}{n}\sum_i
\left[
\hat{r}(x_i,\pi)
+
w_i\left(r_i-\hat{r}(x_i,a_i)\right)
\right],
$$

where

$$
\hat{r}(x,\pi)=\sum_a\pi(a\mid x)\hat{r}(x,a).
$$

For outcome predictions $\hat r(A)=0.4,\hat r(B)=0.6$, the target-model baseline is $0.56$.
The average weighted residual is $(4(0.25)(0.6)+4(-0.6))/5=-0.36$, restoring estimate $0.20$.
Under valid identification and regularity, DR can remain consistent if either nuisance model is correct.
Use independent nuisance fitting or suitable cross-fitting; reusing evaluation outcomes naively can invalidate that claim.

### Leave logged support

**New task:** evaluate a candidate the deterministic logger never exposes.

**Picture and change:** check $\pi(a\mid x)>0\Rightarrow\mu(a\mid x)>0$ before forming ratios.

**Check:** the unsupported table row requires division by zero.
More logs from the same deterministic policy do not reveal its missing reward.
An extrapolating outcome model adds assumptions; calling it doubly robust does not supply observed evidence.

**Use when:** restrict the target policy/population or obtain permitted evidence where coverage is absent.

### Change weight tails

**New task:** reduce sensitivity to a few large weights.

**Picture and change:** compare declared clipping, normalization, nuisance models, and additional supported data separately.

**Check:** clipping at $c=1$ changes the B weight from four to one.
The clipped SNIPS denominator becomes $4(0.25)+1=2$, giving $1/2$ instead of $1/5$.
Lower dispersion therefore need not preserve the estimate.

**Use when:** report ESS, weight quantiles/maxima, and clipping sensitivity rather than trusting nominal sample size.

<!-- learning:core-end -->

<div data-learning-extension>

## Slates and ranking policies

A ranked list is a structured action whose full assignment probability may be very small.
Replacing slate propensities with independent item or position probabilities requires assumptions about interactions and examination.
One click may change whether later items are examined.
An inverse-examination weight is not a generic whole-slate policy-value ratio.
Randomized interleaving or online experiments can answer suitable comparisons when offline identification is inadequate.

## Diagnostics

Before trusting a counterfactual estimate, report:

- how propensities were generated or estimated;
- effective sample size;
- weight quantiles and maximum weight;
- overlap by query and item slice;
- sensitivity to clipping;
- agreement with randomized online results;
- reward maturity and missing outcomes;
- policy and data versions.

A precise result on a supported slice does not identify value for unsupported traffic.
Report target population, reward definition, and unknown outcomes alongside numerical uncertainty.

## In an interview

Does doubly robust mean assumption-free?

<details><summary>Answer</summary>

No. Its nuisance-model robustness operates within a valid identification setup.
It does not fix zero support, interference, confounded assignment, immature rewards, or an ill-defined outcome.

</details>

## Common mistakes

Do not treat the toy log as a confidence interval or empirical reproduction.
More independent supported data can improve precision; more unsupported traffic cannot create an absent counterfactual.
Neither a smaller variance estimate nor a fitted reward model proves examination or assignment assumptions.

</div>

## Practice next

Use this material in [learning-to-rank losses](/concepts/learning-to-rank-losses/), [evaluating a search ranker](/questions/evaluate-search-ranker/), [causal inference for ML decisions](/concepts/causal-inference-for-ml-decisions/), and [personalized search ranking](/guides/personalized-search-ranking/).
