---
title: "How do you choose a loss function?"
description: "Choose a loss from the target, noise model, error cost, and output constraints. A well-trained model still fails when its objective is wrong."
date: "2026-03-09"
draft: false
tags: ["questions"]
category: "questions"
---


Choose a loss by identifying the quantity to predict and separating that estimate from the later deployment decision.

## Problem

**Context.** Three equally weighted outcomes, 0, 0, and 9, represent the intended population for a constant predictor.

**Task.** Find the best constant under squared and absolute error, compare their objective values, and explain the implied prediction targets.
Then distinguish changing a prediction loss from changing the costs of a downstream action.

**Constraints.** Use the same stipulated population and weights for both fits.
Neither loss is universally better. A changed decision threshold does not always require retraining the probability model.

Predict one constant $a$ for three equally weighted outcomes: $0,0,9$.
All three represent the intended population.
Find the best constant under squared error and absolute error.
Then ask whether changing action costs requires a different prediction model.

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>

<p class="visual-kicker">Estimate a quantity, then choose an action</p>
<p class="visual-title">The same observations support different prediction targets</p>

<!-- visual:loss-estimation-vs-decision -->
<figure class="learning-figure" aria-labelledby="loss-comparison-title">
	<h3 class="visual-title" id="loss-comparison-title">A better estimate and a cheaper action are different questions</h3>
	<svg viewBox="0 0 600 700" role="img" aria-labelledby="loss-chart-title loss-chart-desc" style="width: 100%; min-width: 0; height: auto;">
		<title id="loss-chart-title">Loss changes the preferred constant; action costs change the preferred action</title>
		<desc id="loss-chart-desc">For the same outcomes zero, zero and nine, constant zero has absolute residuals zero, zero and nine; constant three has residual magnitudes three, three and six. Squaring then averaging gives MSE 27 versus 18, favoring three. Averaging the distances gives MAE 3 versus 4, favoring zero. Each pair of bars shares a zero baseline and scale; MSE and MAE have different units. In a separate binary task with calibrated positive probability 0.8, false-positive cost 9 and false-negative cost 1, positive action costs 1.8 and negative action costs 0.8. The lower-cost action is negative without changing the probability estimate.</desc>
		<g fill="var(--c-text)" font-family="var(--font-ui)" font-size="24">
			<text x="40" y="30" font-weight="650">1. Same outcomes: 0, 0, 9</text>
			<text x="40" y="76">a = 0: distances 0, 0, 9</text>
			<text x="40" y="112">a = 3: distances 3, 3, 6</text>
			<text x="40" y="162" font-weight="650">MSE: square, then average</text>
			<text x="40" y="209">a = 0</text><rect data-loss="mse" data-value="27" x="150" y="187" width="324" height="28" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)"/><text x="486" y="209">27</text>
			<text x="40" y="255">a = 3</text><rect data-loss="mse" data-value="18" x="150" y="233" width="216" height="28" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)"/><text x="378" y="255">18</text>
			<text x="150" y="290" font-size="22">Minimum at the mean: 3</text>
			<text x="40" y="326" font-weight="650">MAE: average the distances</text>
			<text x="40" y="363">a = 0</text><rect data-loss="mae" data-value="3" x="150" y="341" width="120" height="28" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)"/><text x="282" y="363">3</text>
			<text x="40" y="409">a = 3</text><rect data-loss="mae" data-value="4" x="150" y="387" width="160" height="28" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)"/><text x="322" y="409">4</text>
			<text x="150" y="444" font-size="22">Minimum at the median: 0</text>
			<path d="M40 466H560" stroke="var(--c-rule)"/>
			<text x="40" y="504" font-weight="650">2. Separate binary decision</text>
			<text x="40" y="540" font-size="22">p = 0.8; false positive 9, false negative 1</text>
			<text x="40" y="582">Positive</text><rect data-loss="action-cost" data-value="1.8" x="220" y="560" width="288" height="28" fill="var(--viz-warning-bg)" stroke="var(--viz-warning-stroke)"/><text x="520" y="582">1.8</text>
			<text x="40" y="630">Negative</text><rect data-loss="action-cost" data-value="0.8" x="220" y="608" width="128" height="28" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)"/><text x="360" y="630">0.8</text>
			<text x="40" y="678" font-weight="650">Choose negative: 0.8 &lt; 1.8</text>
		</g>
	</svg>
	<figcaption><strong>Read it this way:</strong> lower bars win within each comparison, not across different losses. Squared distances favor 3; absolute distances favor 0. The separate action example chooses negative despite an unchanged positive probability of 0.8.</figcaption>
</figure>
<p class="diagram-source">References: <a href="https://doi.org/10.1198/016214506000001437">Gneiting and Raftery on proper scoring rules</a>, <a href="https://doi.org/10.1214/aoms/1177703732">Huber on robust location</a>, and <a href="https://scikit-learn.org/stable/modules/classification_threshold.html">scikit-learn's threshold guidance</a>.</p>

## Calculate both targets

For $n$ observations, define $\mathrm{MSE}(a)=n^{-1}\sum_i(a-y_i)^2$ and $\mathrm{MAE}(a)=n^{-1}\sum_i|a-y_i|$.
Here,

$$
\mathrm{MSE}(a)=\frac{a^2+a^2+(a-9)^2}{3},\qquad
\frac{d\,\mathrm{MSE}}{da}=2a-6.
$$

Squared error selects $a=3$, the mean. Absolute error selects $a=0$, the median.

| Constant | MSE | MAE |
| --- | --- | --- |
| $0$ | $27$ | $3$ |
| $3$ | $18$ | $4$ |

If nine is a genuine costly outcome, discarding it as an outlier changes the task.
At each input, squared and absolute loss similarly target conditional means and medians when their risks exist.

## Change the requirement

### 1. Limit contamination influence

**New task:** nine may be corrupted, and the required output is a robust location.

**Picture and change:** Huber loss bounds each residual's gradient.
For residual $r=a-y$ and scale $\delta>0$,

$$
H_\delta(r)=\begin{cases}
r^2/2,&|r|\le\delta,\\
\delta(|r|-\delta/2),&|r|>\delta.
\end{cases}
$$

**Check:** with $\delta=1$ at $a=0$, the nine observation contributes gradient $-1$, not squared-error gradient $-18$.
The mean Huber loss here is minimized at $a=0.5$ because $2a-1=0$.

**Use when:** contamination may cause large residuals, rather than a genuine tail needed for mean estimation.

### 2. Change action costs

**New task:** false positives cost $9$, false negatives cost $1$, and the calibrated positive probability is $p=0.8$.
Correct decisions have zero cost.

**Picture and change:** compare the expected costs of both actions before retraining.
Choose positive when

$$
C_{\mathrm{FP}}(1-p)<C_{\mathrm{FN}}p
\quad\Longleftrightarrow\quad
p>\frac{C_{\mathrm{FP}}}{C_{\mathrm{FP}}+C_{\mathrm{FN}}}.
$$

**Check:** the threshold is $0.9$.
Positive action costs $1.8$ in expectation; negative costs $0.8$. Choose negative.

**Use when:** probabilities remain suitable but the cost of acting changes.
Capacity and other constraints may require a different decision rule.

## Do not treat every score as a probability

Weighted or focal objectives can change score meaning.
Check calibration before applying a threshold derived for true probabilities.
Focal modulation emphasizes difficult examples; it does not specify which error costs more.

## Try increasing the precision requirement

The model already ranks relevant cases well. What changes first?

<details>
<summary>Check your answer</summary>

Test thresholds on representative validation data, separate from final evaluation.
Report recall and accepted volume. Retrain only if the resulting operating choices fail the requirement.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<span id="what-an-l6-answer-sounds-like"></span>
<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Likelihood and output meaning

Negative log likelihood is $-\log p_\theta(y\mid x)$.
Fixed-variance Gaussian likelihood gives squared error up to constants and scale.
Fixed-scale Laplace likelihood gives absolute error. Learned scales require their log terms too.
These likelihood assumptions are not required merely to target means or medians.

Categorical cross-entropy is $-\sum_c y_c\log p_c$ for one-hot targets.
Squared error on probability vectors can also be a proper Brier score.
Gradient problems depend on the output link and parameterization, not just the loss name.

Focal loss multiplies log loss by $(1-p_t)^\gamma$ for the true-class probability $p_t$.
Optional class weights are separate from that modulation; see the [original focal-loss paper](https://arxiv.org/abs/1708.02002).

## Common follow-up

For a higher precision target, thresholding may suffice.
Other options include cost-sensitive retraining, a ranking surrogate, or high-recall retrieval followed by a precision-focused filter.
Evaluate representative held-out precision, recall, volume and segment behavior.

## Combined objectives

Ranking losses target order; contrastive and triplet losses shape embedding relationships.
Multi-task weights change the combined objective.
[Uncertainty weighting](https://arxiv.org/abs/1705.07115) and [GradNorm](https://arxiv.org/abs/1711.02257) use different weight-selection rules.
Auxiliary reconstruction or load-balancing losses also change the optimized objective, even when the main term stays unchanged.
Next-token, preference and reward-based training objectives need downstream behavioral evaluation rather than assuming equal loss reductions mean equal improvements.

Related: [multi-task learning](/concepts/multi-task-learning-objective-interference/), [cross-entropy and softmax](/concepts/cross-entropy-softmax/), [regularization](/concepts/regularization/), and [calibration](/concepts/calibration/).

</div>
