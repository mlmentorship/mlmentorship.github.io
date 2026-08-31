---
title: "Epistemic vs aleatoric uncertainty"
description: "Epistemic uncertainty shrinks with more data; aleatoric uncertainty does not. Confusing them causes miscalibration and wasted data collection."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Aleatoric uncertainty** is irreducible noise in the data-generating process. **Epistemic uncertainty** is uncertainty about the model itself, due to limited data. More training data shrinks epistemic uncertainty but leaves aleatoric uncertainty unchanged.

Decision-making under uncertainty depends on which kind you have:

- **High aleatoric, low epistemic**: the model knows the world well, but the world is noisy. Collect more diverse features, not more samples. A coin flip has 0.5 aleatoric uncertainty no matter how many flips you observe.
- **Low aleatoric, high epistemic**: the model is uncertain because it has not seen this region of input space. Active learning, more data, ensemble disagreement signals.
- **Both high**: the world is noisy and you have not modeled it well. Both data collection and model improvement help.

Production ML systems that report a single "confidence" number conflate the two and make incorrect downstream decisions: refusing to predict when the world is just noisy, or being overconfident in regions the model has never seen.

## How they show up mathematically

For a Bayesian predictive distribution $p(y \mid x, D) = \int p(y \mid x, \theta) p(\theta \mid D) \, d\theta$:

- **Aleatoric**: spread within $p(y \mid x, \theta)$ for a fixed $\theta$.
- **Epistemic**: spread of the conditional means $\mathbb{E}[y \mid x, \theta]$ over the posterior $p(\theta \mid D)$.

For regression with a Gaussian likelihood, total variance decomposes as

$$
\mathrm{Var}[y \mid x, D] = \underbrace{\mathbb{E}_\theta[\sigma^2(x; \theta)]}_{\text{aleatoric}} + \underbrace{\mathrm{Var}_\theta[\mu(x; \theta)]}_{\text{epistemic}}.
$$

A clean separation: aleatoric is the average noise prediction; epistemic is the disagreement between models.

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
	<figcaption><strong>Read it this way:</strong> each line is one model draw's predictive range; each diamond is that model's mean. On the left, individual models are uncertain even though their means agree, so the spread is aleatoric. On the right, each model is confident but their means disagree, so the spread is epistemic. Both panels cover the same overall output envelope, showing why one aggregate “confidence” cannot identify the remedy. This is an original construction based on the law of total variance and Kendall and Gal (2017).</figcaption>
</figure>

## Estimating each in practice

### Aleatoric

Predict the noise directly. For regression, output both mean and log-variance: $\mu(x), \log \sigma^2(x)$. Train with the Gaussian negative log-likelihood ([Kendall & Gal, 2017](https://arxiv.org/abs/1703.04977)):

$$
\mathcal{L} = \frac{1}{2} \log \sigma^2(x) + \frac{(y - \mu(x))^2}{2 \sigma^2(x)}.
$$

The model learns to predict where the data is noisy (large $\sigma^2$) and where it is clean (small $\sigma^2$). For classification, the predicted probabilities themselves encode aleatoric uncertainty.

### Epistemic

Capture model uncertainty. Three practical approaches:

| Approach | What it does | Cost |
|---|---|---|
| **Deep ensembles** ([Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474)) | Train $N$ independent models, look at disagreement | $N$x training cost |
| **MC dropout** ([Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)) | Keep dropout active at inference, take samples | $N$x inference cost |
| **Variational Bayes / SWAG / Laplace approx.** | Approximate $p(\theta \mid D)$ with a tractable distribution | Training overhead |

Deep ensembles are the strongest and simplest; MC dropout is cheaper but a less faithful posterior approximation. Both produce $N$ predictions whose disagreement is the epistemic estimate.

### Decomposition

For classification with a deep ensemble of $M$ models producing probabilities $p_m(y \mid x)$:

$$
H[\bar{p}] = \underbrace{\frac{1}{M} \sum_m H[p_m]}_{\text{aleatoric}} + \underbrace{H[\bar{p}] - \frac{1}{M} \sum_m H[p_m]}_{\text{epistemic}},
$$

where $\bar{p} = \frac{1}{M} \sum_m p_m$ is the ensemble average. Predictive entropy splits into within-model (aleatoric) and between-model (epistemic) components.

## Where each matters in practice

- **Active learning**: query labels for inputs with highest epistemic uncertainty. Aleatoric uncertainty is irreducible, so labeling a noisy input wastes budget.
- **Out-of-distribution detection**: high epistemic uncertainty signals OOD; high aleatoric does not.
- **Safe decision-making**: medical, autonomous driving. High epistemic uncertainty should trigger "abstain or fall back to human"; high aleatoric uncertainty should still produce a calibrated prediction.
- **Reinforcement learning**: epistemic uncertainty drives exploration (UCB, RND); aleatoric is just reward noise.

## Common pitfalls

- **Reporting "confidence" without saying which kind.** A single number conflates the two.
- **Treating softmax probabilities as a measure of uncertainty.** They estimate aleatoric uncertainty but say nothing about epistemic uncertainty. A confidently wrong prediction far from training data has low entropy and high epistemic uncertainty.
- **Using MC dropout in deployment without retraining the model with dropout active.** Dropout-as-Bayes only works if the model was trained with dropout in the same way.
- **Comparing ensembles of size 2 to "Bayesian deep learning."** Ensembles need 5+ members to capture meaningful posterior spread.

## Related

- [Expected Calibration Error](/concepts/expected-calibration-error/).
- [Calibration](/concepts/calibration/).
- [Bayes' rule and the posterior](/concepts/bayes-rule-and-posterior/).
