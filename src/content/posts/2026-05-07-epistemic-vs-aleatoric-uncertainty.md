---
title: "Epistemic vs aleatoric uncertainty"
description: "Decompose a specified predictive mixture into within-model variation and disagreement, then test what new inputs, data and categorical outputs change."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

Predictive uncertainty separates within-model variation from disagreement only relative to the chosen inputs, likelihood and model distribution.

## Summary

Which proposed collection action can reduce the predictive spread at this input?
Fix input $x$, observed dataset $D$ and a distribution over models indexed by $m$.
Use the hypothetical equal-weight mixture $Y\mid m=1\sim N(0,1)$ and $Y\mid m=2\sim N(2,1)$.
Each normal's second parameter is its variance.

| Quantity | Calculation | Result |
| --- | --- | --- |
| Predictive mean | $(0+2)/2$ | 1 |
| Average conditional variance | $(1+1)/2$ | 1 |
| Variance of conditional means | $((0-1)^2+(2-1)^2)/2$ | 1 |
| Total variance | $1+1$ | 2 |

Independently, $E[Y^2]=((1+0^2)+(1+2^2))/2=3$.
Subtracting the squared mean gives $3-1=2$ again.

## How they show up mathematically

Let $\mu_m=E[Y\mid x,m]$ and $\mu=E_m[\mu_m]$; assume finite second moments.
Write $Y-\mu=(Y-\mu_m)+(\mu_m-\mu)$ and square.
The cross term vanishes after conditioning on $m$, because $E[Y-\mu_m\mid x,m]=0$.
Thus, under the specified model mixture,

$$
\operatorname{Var}(Y\mid x,D)=E_m[\operatorname{Var}(Y\mid x,m)]+\operatorname{Var}_m(\mu_m).
$$

Call these model-relative within and between terms aleatoric and epistemic, respectively.
Both use squared output units.
The identity does not require Gaussian components; their normal form only simplifies this example.
More representative data can change model uncertainty, but no universal monotonic decrease is guaranteed.

<!-- visual:uncertainty-within-between-models -->
<figure class="learning-figure" aria-labelledby="uncertainty-decomposition-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="uncertainty-decomposition-title">Does predictive spread live within models or between them?</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison of aleatoric and epistemic predictive spread across three posterior model draws">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 250" role="img" aria-labelledby="aleatoric-panel-title aleatoric-panel-desc">
				<title id="aleatoric-panel-title">Mostly aleatoric uncertainty</title>
				<desc id="aleatoric-panel-desc">Three posterior model draws theta one, theta two, and theta three have broad predictive ranges with diamond means aligned at the same output value. Each model is individually uncertain, but the models agree with one another. Additional examples of the same kind do not remove this outcome noise.</desc>
				<rect class="viz-plot-bg" x="8" y="27" width="284" height="214" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="17">MOSTLY ALEATORIC</text>
				<path class="viz-axis" d="M42 205H270"></path>
				<g class="viz-label" text-anchor="middle"><text x="55" y="222">low y</text><text x="258" y="222">high y</text></g>
				<g class="viz-axis-label"><text x="16" y="74">θ1</text><text x="16" y="124">θ2</text><text x="16" y="174">θ3</text></g>
				<g style="fill:none;stroke:var(--viz-input-stroke);stroke-width:5;stroke-linecap:round">
					<path d="M55 70H255"></path><path d="M55 120H255"></path><path d="M55 170H255"></path>
				</g>
				<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2">
					<path d="M155 63L162 70L155 77L148 70Z"></path><path d="M155 113L162 120L155 127L148 120Z"></path><path d="M155 163L162 170L155 177L148 170Z"></path>
				</g>
				<path class="viz-operating-guide" d="M155 48V190"></path>
				<text class="viz-callout" x="155" y="43" text-anchor="middle">means agree</text>
				<text class="viz-label" x="155" y="237" text-anchor="middle">wide within-model ranges</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 250" role="img" aria-labelledby="epistemic-panel-title epistemic-panel-desc">
				<title id="epistemic-panel-title">Mostly epistemic uncertainty</title>
				<desc id="epistemic-panel-desc">Three posterior model draws have narrow predictive ranges, but their diamond means are separated across low, middle, and high output values. Each model is individually confident, yet the models disagree. More representative data can concentrate the posterior and bring these model predictions together.</desc>
				<rect class="viz-plot-bg" x="8" y="27" width="284" height="214" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="17">MOSTLY EPISTEMIC</text>
				<path class="viz-axis" d="M42 205H270"></path>
				<g class="viz-label" text-anchor="middle"><text x="55" y="222">low y</text><text x="258" y="222">high y</text></g>
				<g class="viz-axis-label"><text x="16" y="74">θ1</text><text x="16" y="124">θ2</text><text x="16" y="174">θ3</text></g>
				<g style="fill:none;stroke:var(--viz-input-stroke);stroke-width:5;stroke-linecap:round">
					<path d="M55 70H105"></path><path d="M130 120H180"></path><path d="M205 170H255"></path>
				</g>
				<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2">
					<path d="M80 63L87 70L80 77L73 70Z"></path><path d="M155 113L162 120L155 127L148 120Z"></path><path d="M230 163L237 170L230 177L223 170Z"></path>
				</g>
				<path d="M80 43H230" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-dasharray:4 3"></path>
				<text class="viz-callout" x="155" y="38" text-anchor="middle">means disagree</text>
				<text class="viz-label" x="155" y="237" text-anchor="middle">narrow within-model ranges</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> lines show qualitative conditional ranges; diamonds show conditional means. Compare range width with mean separation. Equal endpoint envelopes do not imply equal variances or coverage. These are schematic model draws, not measured posterior adequacy. Original construction based on total variance and Kendall and Gal (2017). <span class="visual-reader-check" data-reader-check="uncertainty-within-between-models"><strong>Check:</strong> Which uncertainty term grows when model means separate while each model's predictive spread stays narrow?</span></figcaption>
</figure>

## Change the information or output

### Add an informative sensor

**New task:** supply missing predictive information, rather than repeat the same observations.

**Picture and change:** suppose each conditional variance falls from one to $0.25$, while means and mixture weights stay fixed.

**Check:** within variance becomes $0.25$; between variance remains one; total variance is $1.25$.
The old within-model noise was not irreducible relative to the expanded inputs.

**Use when:** the proposed sensor changes the conditioning information; the stated reduction still needs empirical validation.

### Predict a binary label

**New task:** measure information spread for equally weighted probabilities $(0.9,0.1)$ and $(0.1,0.9)$.

**Picture and change:** their mean is $(0.5,0.5)$.
Use entropy in bits, not regression variance.

**Check:** predictive entropy is one bit; average conditional entropy is approximately $0.468996$ bits.
The difference is approximately $0.531004$ bits of mutual information between the label and this mixture's model index.

**Use when:** categorical predictions make an entropy decomposition appropriate.
A diverse ensemble is not automatically an exact posterior.

## Failure and transfer

All members can predict mean zero and variance $0.01$ while target outputs are near ten.
Zero disagreement does not certify correctness, calibration or in-distribution inputs.
Now change the Gaussian weights to three quarters and one quarter. What happens?

<details>
<summary>Reasoned answer</summary>

The predictive mean is $0.5$.
Between variance is $0.75$, within variance remains one, and total variance is $1.75$.
Changing weights changes the distribution being averaged, not merely an uncertainty label.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Estimating each in practice

### Aleatoric

For heteroskedastic Gaussian regression, predict mean $\mu(x)$ and log variance $s(x)$, making variance $e^{s(x)}$ positive.
Ignoring the normalizing constant, NLL is $[s(x)+(y-\mu(x))^2e^{-s(x)}]/2$.
Fit quality and likelihood assumptions determine whether this estimates actual conditional noise.
See <a href="https://arxiv.org/abs/1703.04977">Kendall and Gal</a>.

### Epistemic

| Approach | Approximation | Cost to assess |
| --- | --- | --- |
| Deep ensemble | diversity across separately fitted predictors | multiple training and serving runs |
| MC dropout | stochastic masks in a compatible dropout-trained model | repeated forward passes |
| Variational, SWAG or Laplace | a specified approximate parameter distribution | fitting, storage and sampling overhead |

Measure approximation adequacy instead of declaring a universal strongest method or minimum ensemble size.
BatchNorm behavior must remain controlled when enabling dropout sampling.
References: <a href="https://arxiv.org/abs/1612.01474">deep ensembles</a> and <a href="https://arxiv.org/abs/1506.02142">MC dropout</a>.

### Decomposition

For discrete mixture weights $w_m$, let $p=\sum_m w_mp_m$.
Then $H(p)=\sum_m w_mH(p_m)+I(Y;M)$.
The difference is model-index mutual information under that joint distribution, not an automatic measurement of real-world ignorance.

## Where each matters in practice

Active learning weighs expected information or decision benefit against acquisition cost; noisy labels can still be useful.
Abstention, fallback and exploration require task-specific action costs and target-domain validation.
No decomposition provides a universal out-of-distribution detector or safety guarantee.

## Common pitfalls

One softmax vector cannot identify how uncertainty divides across plausible models.
Agreement can reflect common misspecification.
A fair binary outcome has variance $0.25$ or entropy one bit; probability $0.5$ alone does not specify uncertainty units.

## Related

- [Expected Calibration Error](/concepts/expected-calibration-error/).
- [Calibration](/concepts/calibration/).
- [Bayes' rule and the posterior](/concepts/bayes-rule-and-posterior/).

</div>
