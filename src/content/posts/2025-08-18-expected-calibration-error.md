---
title: "Expected Calibration Error (ECE)"
description: "How well do predicted probabilities match empirical frequencies? Bin predictions by confidence, compare bin-mean confidence to bin-accuracy."
date: "2025-08-18"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Expected Calibration Error** measures how well a model's predicted probabilities match empirical accuracies. Bin predictions by predicted confidence, and compute the weighted average of $|\text{accuracy in bin} - \text{average confidence in bin}|$:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \cdot \big| \text{acc}(B_m) - \text{conf}(B_m) \big|.
$$

A classifier that scores well on accuracy can still produce wildly miscalibrated probabilities. Predicting "90% confident" when only 60% of such predictions are correct. Calibration matters whenever:

- The probability is used downstream (decision thresholds, expected-cost calculations, risk scoring).
- A human reads the probability (medical diagnosis, fraud alerts).
- The model is combined with other signals (Bayesian fusion).

ECE is the standard single-number calibration metric.

## The mechanism

For a classifier producing a predicted label $\hat y_i$ and its confidence $c_i \in [0, 1]$ on each of $N$ examples with true labels $y_i$:

1. **Bin** the predictions by confidence. Standard: $M = 10$ equal-width bins covering $[0, 1]$.
2. For each bin $B_m$:
   - $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} c_i$ (average predicted-class confidence).
   - $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}[\hat y_i = y_i]$ (empirical accuracy).
3. **Aggregate**: ECE = weighted average of bin gaps.

Sample ECE is zero when each populated bin's empirical accuracy equals its average confidence.
Finite-sample noise and binning mean this does not prove population calibration.
Conversely, a calibrated model need not have zero empirical ECE on a finite sample.
The worked example below is multiclass, so predicted-class confidence can be below 50%.

<!-- visual:ece-weighted-bin-contributions -->
<figure class="learning-figure plot-panel" aria-labelledby="ece-aggregation-visual-title">
	<p class="visual-kicker">Worked aggregation</p>
	<p class="visual-title" id="ece-aggregation-visual-title">A bin's ECE contribution depends on both its population and its calibration gap.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 300" role="img" aria-labelledby="ece-aggregation-svg-title ece-aggregation-svg-desc">
			<title id="ece-aggregation-svg-title">Three confidence bins contributing to a 13-point expected calibration error</title>
			<desc id="ece-aggregation-svg-desc">An original example has 100 predictions in three bins. Bin A contains 50 examples, has 30 percent confidence and 20 percent accuracy, so its 10-point gap weighted by 50 percent contributes 5 ECE points. Bin B contains 40 examples, has 70 percent confidence and 60 percent accuracy, so its 10-point gap weighted by 40 percent contributes 4 points. Bin C contains 10 examples, has 90 percent confidence and 50 percent accuracy, so its 40-point gap weighted by 10 percent also contributes 4 points. The contributions sum to 13 percentage points. Population bars and gap bars use separate common scales, showing that Bin C's largest gap is offset by its smallest population.</desc>
			<text class="viz-axis-label" x="14" y="20">BIN</text>
			<text class="viz-axis-label" x="72" y="20">SHARE OF 100</text>
			<text class="viz-axis-label" x="185" y="20">ABSOLUTE GAP</text>
			<text class="viz-axis-label" x="292" y="20">ECE</text>
			<path class="viz-gridline" d="M8 30H352 M8 96H352 M8 162H352 M8 228H352"></path>
			<text class="viz-callout" x="14" y="57">A</text>
			<text class="viz-label" x="14" y="75">30% → 20%</text>
			<rect class="viz-plot-bg" x="72" y="42" width="88" height="18" rx="2"></rect>
			<rect class="viz-node--input" x="72" y="42" width="44" height="18" rx="2"></rect>
			<text class="viz-callout" x="72" y="78">50/100</text>
			<rect class="viz-plot-bg" x="185" y="42" width="80" height="18" rx="2"></rect>
			<rect class="viz-node--focus" x="185" y="42" width="20" height="18" rx="2"></rect>
			<text class="viz-callout" x="185" y="78">|20−30| = 10%</text>
			<text class="viz-callout" x="292" y="55">5 pts</text>
			<text class="viz-label" x="292" y="75">.50 × .10</text>
			<text class="viz-callout" x="14" y="123">B</text>
			<text class="viz-label" x="14" y="141">70% → 60%</text>
			<rect class="viz-plot-bg" x="72" y="108" width="88" height="18" rx="2"></rect>
			<rect class="viz-node--input" x="72" y="108" width="35.2" height="18" rx="2"></rect>
			<text class="viz-callout" x="72" y="144">40/100</text>
			<rect class="viz-plot-bg" x="185" y="108" width="80" height="18" rx="2"></rect>
			<rect class="viz-node--focus" x="185" y="108" width="20" height="18" rx="2"></rect>
			<text class="viz-callout" x="185" y="144">|60−70| = 10%</text>
			<text class="viz-callout" x="292" y="121">4 pts</text>
			<text class="viz-label" x="292" y="141">.40 × .10</text>
			<text class="viz-callout" x="14" y="189">C</text>
			<text class="viz-label" x="14" y="207">90% → 50%</text>
			<rect class="viz-plot-bg" x="72" y="174" width="88" height="18" rx="2"></rect>
			<rect class="viz-node--input" x="72" y="174" width="8.8" height="18" rx="2"></rect>
			<text class="viz-callout" x="72" y="210">10/100</text>
			<rect class="viz-plot-bg" x="185" y="174" width="80" height="18" rx="2"></rect>
			<rect class="viz-node--focus" x="185" y="174" width="80" height="18" rx="2"></rect>
			<text class="viz-callout" x="185" y="210">|50−90| = 40%</text>
			<text class="viz-callout" x="292" y="187">4 pts</text>
			<text class="viz-label" x="292" y="207">.10 × .40</text>
			<path class="viz-axis" d="M72 241H160 M72 237V245 M116 237V245 M160 237V245 M185 241H265 M185 237V245 M225 237V245 M265 237V245"></path>
			<text class="viz-label" x="72" y="258">0</text>
			<text class="viz-label" x="109" y="258">50%</text>
			<text class="viz-label" x="151" y="258">100%</text>
			<text class="viz-label" x="185" y="258">0</text>
			<text class="viz-label" x="218" y="258">20</text>
			<text class="viz-label" x="255" y="258">40 pts</text>
			<text class="viz-callout" x="14" y="287">ECE = 5 + 4 + 4 = 13 percentage points</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> multiply across each row, then add down the ECE column. Bin C has the largest gap, but only 10% of examples, so it contributes the same four points as the denser Bin B. This original example follows the ECE definition in <a href="https://doi.org/10.1609/aaai.v29i1.9602">Naeini et al. (2015)</a> and <a href="https://proceedings.mlr.press/v70/guo17a.html">Guo et al. (2017)</a>. <span class="visual-reader-check" data-reader-check="ece-weighted-bin-contributions"><strong>Check:</strong> Why does C's 40-point gap contribute the same four ECE points as B's ten-point gap?</span></figcaption>
</figure>

## Reliability diagram

The visual companion to ECE: plot bin accuracy vs. bin confidence. Perfect calibration is the diagonal $y = x$. Above-diagonal: under-confident. Below-diagonal: over-confident (the typical deep-net failure).

Always plot the reliability diagram alongside reporting ECE. Single ECE number can hide dramatic per-bin issues.

## Variants

- **Maximum Calibration Error (MCE)**: $\max_m |\text{acc}(B_m) - \text{conf}(B_m)|$. Worst-case bin gap.
- **Adaptive ECE**: equal-frequency bins instead of equal-width. Stable when predictions concentrate near 0 or 1.
- **Class-wise ECE**: per-class calibration; matters in multi-class.
- **Top-label ECE** (multi-class): compute ECE on the predicted-class probability only.

## Why deep nets are miscalibrated

Modern neural networks [(Guo et al., 2017)](https://arxiv.org/abs/1706.04599) are typically **overconfident**:

- Guo et al. observed overconfidence in the classification models and datasets they studied, not every architecture or deployment population.
- Cross-entropy is a proper probability objective; fitting it on finite training data does not guarantee calibrated deployment probabilities.
- Regularization, model fitting, sample size, and distribution shift can affect calibration. Measure their effect rather than assuming an architecture ordering.

## Calibration methods

Fit a post-hoc map on calibration data, then evaluate the frozen map on separate, representative cases.
The [calibration lesson](/concepts/calibration/) derives the maps and their effects on downstream decisions.

- **Temperature scaling** [(Guo et al., 2017)](https://arxiv.org/abs/1706.04599): divide every class logit by one learned scalar $T>0$. Values above one soften probabilities; values below one sharpen them. Shared positive scaling preserves multiclass argmax, not every probability-threshold action.
- **Platt scaling**: fit a binary logistic map with a slope and intercept. A shifted intercept can change decisions at probability 0.5.
- **Isotonic regression**: fit a nondecreasing map. Plateaus can create ties or move scores across an operating threshold; flexible maps need enough calibration data.
- **Vector / matrix scaling**: use class-specific scales/biases or a full affine logit map. These can change class ordering and accuracy.

Calibration is not generally lossless on accuracy or decision cost.
Evaluate probability quality and the actual decision rule, including relevant slices, after fitting the map.

**Predict before revealing:** logits `[2,1]` choose class 0. Which class wins after shared temperature 10, versus class-specific temperatures `[10,1]`?

<details><summary>Check the ordering and the action</summary>

Shared scaling gives `[0.2,0.1]`, so class 0 still wins.
Class-specific scaling gives `[0.2,1]`, so class 1 wins.
Even shared scaling changes probabilities, so a cost-based threshold can trigger a different action without changing argmax.

</details>

## Limits of ECE

- **Binning artifact**: ECE depends on bin choice. Adaptive ECE is more stable.
- **Confidence vs. probability of correctness**: ECE on multi-class typically uses only the top-class probability, ignoring whether the full distribution is well-calibrated.
- **Doesn't measure sharpness**: a model that predicts $0.5$ for everything has ECE = 0 if base rate is $0.5$, but is useless. Combine ECE with proper scoring rules (Brier score, log loss).

## Common pitfalls

- **Fitting and evaluating on the same cases.** Keep calibration fitting separate from final evaluation; retest after population changes.
- **Reporting ECE without a reliability diagram.** Visual asymmetries can be invisible in the single number.
- **Assuming rescaling must help.** Compare the frozen map with the uncalibrated baseline on held-out probability and decision metrics.
- **Confusing calibration with accuracy.** A 95%-accuracy model with ECE 0.20 is still untrustworthy whenever the probability matters.

## Related

- [Calibration](/concepts/calibration/). Broader treatment of calibration methods.
- [Confusion matrix](/concepts/confusion-matrix-and-classification-metrics/). Accuracy-based metrics.
