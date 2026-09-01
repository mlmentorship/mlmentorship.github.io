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

A click is evidence about both relevance and exposure. Items near the top are examined more often, so raw click rate rewards the logging ranker and its positions.

Counterfactual learning and evaluation correct for known assignment or examination probabilities. Inverse propensity scoring reweights observed outcomes, self-normalization can reduce instability, and doubly robust estimators combine weighting with an outcome model. Every method still needs overlap between logged actions and the policy being evaluated.

## The observation process

A simple examination model writes click probability as

$$
P(C=1\mid q,d,k)
=
P(E=1\mid k)P(R=1\mid q,d),
$$

where $E$ means the user examined position $k$ and $R$ means document $d$ was relevant to query $q$.

This factorization is an assumption. Trust, snippets, neighboring results, device type, and query intent can make examination depend on more than position.

The useful lesson remains: no click can mean either “seen and rejected” or “not seen.” Raw logs do not identify which case occurred.

<!-- visual:position-bias-inverse-examination-weight -->
<figure class="learning-figure" aria-labelledby="position-bias-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="position-bias-visual-title">Separate relevance from exposure, then trace how inverse examination weighting restores equal expected signal.</p>
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
	<figcaption><strong>Read it this way:</strong> compare “Expected clicks” first: identical relevance produces 60 clicks at position 1 but only 15 at position 4 because fewer users examine the lower result. Then apply the inverse examination weight: each lower-position click represents four exposure opportunities, so both positions contribute 60 in expectation. This is an illustrative calculation under the stated position-based examination model, not a claim that real users examine position 4 exactly 25% of the time. Original example checked against <a href="https://arxiv.org/abs/1608.04468">Joachims, Swaminathan, and Schnabel’s counterfactual LTR formulation</a>.</figcaption>
</figure>

## Estimate propensities

A propensity is the probability that the logging process creates an observation used by the estimator.

Two propensities appear in ranking work:

- **examination propensity:** probability that a position is examined;
- **policy propensity:** probability that the logging policy chooses an item or slate in a context.

Estimate examination effects with randomized swaps, interventions on result order, or a validated click model. Log policy propensities directly when traffic uses randomized exploration.

A deterministic production ranker has no support for actions it never chooses. Historical logs from that policy cannot evaluate arbitrary new rankings without additional assumptions or exploration.

## Inverse propensity scoring

For contextual bandit data $(x_i,a_i,r_i)$ logged under policy $\mu$, the value of target policy $\pi$ can be estimated by

$$
\widehat{V}_{IPS}(\pi)
=
\frac{1}{n}\sum_{i=1}^n
\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}r_i.
$$

The ratio gives more weight to actions that the target policy favors but the logging policy chose rarely.

Under correct logged propensities, overlap, and stable outcomes, IPS is unbiased. It can have very high variance when $\mu(a_i\mid x_i)$ is small.

For position-debiased training, inverse examination propensity can weight clicked examples. The exact estimator depends on the click model and loss. State which observation process the weight corrects.

## Self-normalization and clipping

Self-normalized IPS, often called SNIPS, divides by the sum of weights:

$$
\widehat{V}_{SNIPS}(\pi)
=
\frac{\sum_i w_i r_i}{\sum_i w_i},
\qquad
w_i=\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}.
$$

Self-normalization often lowers variance but introduces finite-sample bias.

Weight clipping applies

$$
\tilde{w}_i=\min(w_i,c).
$$

It prevents a few rare actions from dominating the estimate. Clipping also adds bias. Report the threshold and show sensitivity to it. SNIPS and clipping are separate choices and may be used together.

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

The outcome model supplies a baseline, and the weighted residual corrects it on logged actions. Under standard assumptions, the estimator can remain consistent if either the propensity model or outcome model is correct.

It does not fix missing support, interference, bad reward definitions, or hidden confounding in a nonrandom logging process.

## Slates and ranking policies

A ranked list is a structured action. The probability of a complete slate may be tiny, which makes full-slate IPS impractical.

Common simplifications model positions or item choices separately. These require assumptions about interaction among results. A click on one item can change whether later items are examined, so independent-position models may be wrong.

Use randomized interleaving or online A/B tests when reliable offline identification would require unrealistic slate assumptions.

## Worked example

A logging policy places item A first with probability 0.8 and item B first with probability 0.2. The target policy reverses those probabilities.

When B appears first, its target-to-logging weight is $0.8/0.2=4$. When A appears first, its weight is $0.2/0.8=0.25$.

The large weight for B makes the estimate sensitive to a small number of B observations. More exploration, a larger sample, clipping, or a useful outcome model can reduce variance.

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

A low-variance estimate for a narrow supported population may be more useful than a nominal estimate for unsupported traffic.

## In an interview

Use this order:

1. Separate exposure from relevance.
2. State the logging and target policies.
3. Identify the propensity being used.
4. Write the IPS ratio and support condition.
5. Explain SNIPS, clipping, and doubly robust estimation.
6. Discuss slate structure, variance, and online validation.

## Common mistakes

- Calling raw clicks relevance labels.
- Treating examination propensity and policy propensity as the same quantity.
- Saying SNIPS is weight clipping.
- Ignoring actions with zero logging probability.
- Estimating propensities from outcomes without an assignment model.
- Reporting IPS without weight diagnostics.
- Assuming doubly robust means assumption-free.

## Practice next

Use this material in [learning-to-rank losses](/concepts/learning-to-rank-losses/), [evaluating a search ranker](/questions/evaluate-search-ranker/), [causal inference for ML decisions](/concepts/causal-inference-for-ml-decisions/), and [personalized search ranking](/guides/personalized-search-ranking/).
