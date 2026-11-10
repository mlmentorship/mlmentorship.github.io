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
	<h3 class="visual-title" id="loss-comparison-title">Choose the estimation target first; choose the action from suitable probabilities and actual costs</h3><div class="visual-argument" data-visual-argument="loss-estimation-vs-decision" data-argument-kind="bars" data-argument-composition="estimation-and-decision"><svg viewBox="0 0 360 2700" role="img" aria-label="Choose the estimation target first; choose the action from suitable probabilities and actual costs. Squared loss selects mean three; absolute loss selects median zero for the same population. Neither target is universally better. Robust location requires a contamination assumption, not automatic deletion of a genuine tail. Separately, a suitable calibrated probability can stay fixed while changed action costs reverse the decision. Validate thresholds, recall and volume. Weighted or focal scores are not automatically calibrated probabilities."><title>Choose the estimation target first; choose the action from suitable probabilities and actual costs</title><desc>Squared loss selects mean three; absolute loss selects median zero for the same population. Neither target is universally better. Robust location requires a contamination assumption, not automatic deletion of a genuine tail. Separately, a suitable calibrated probability can stay fixed while changed action costs reverse the decision. Validate thresholds, recall and volume. Weighted or focal scores are not automatically calibrated probabilities.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same population, different estimation</tspan><tspan x="20" dy="18">targets</tspan></text><text x="20" y="110" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Zero, zero, nine; equal weights. Fit</tspan><tspan x="20" dy="18">one constant. Square areas show</tspan><tspan x="20" dy="18">squared error; short bars show</tspan><tspan x="20" dy="18">absolute error.</tspan></text><text x="20" y="300" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Constant zero: median target</tspan></text><text x="65" y="365" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="65" dy="0">Target 0</tspan></text><circle cx="65" cy="395" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="65" y="490" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="65" dy="0">0</tspan></text><path d="M30 535L100 535" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="65" y="575" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="65" dy="0">0</tspan></text><text x="180" y="365" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Target 0</tspan></text><circle cx="180" cy="395" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="180" y="490" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="180" dy="0">0</tspan></text><path d="M145 535L215 535" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="575" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="180" dy="0">0</tspan></text><text x="295" y="365" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="295" dy="0">Target 9</tspan></text><rect x="263.5" y="395" width="63" height="63" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="295" y="490" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="295" dy="0">81</tspan></text><path d="M260 535L330 535" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect x="260" y="528" width="70" height="14" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="295" y="575" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="295" dy="0">9</tspan></text><text x="20" y="620" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">Mean squared error: 27. Mean absolute</tspan><tspan x="20" dy="18">error: 3.</tspan></text><text x="20" y="680" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Constant three: mean target</tspan></text><text x="65" y="745" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="65" dy="0">Target 0</tspan></text><rect x="54.5" y="775" width="21" height="21" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="65" y="870" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="65" dy="0">9</tspan></text><path d="M30 915L100 915" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect x="30" y="908" width="23.333333333333332" height="14" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="65" y="955" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="65" dy="0">3</tspan></text><text x="180" y="745" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Target 0</tspan></text><rect x="169.5" y="775" width="21" height="21" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="180" y="870" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="180" dy="0">9</tspan></text><path d="M145 915L215 915" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect x="145" y="908" width="23.333333333333332" height="14" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="180" y="955" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="180" dy="0">3</tspan></text><text x="295" y="745" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="295" dy="0">Target 9</tspan></text><rect x="274" y="775" width="42" height="42" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="295" y="870" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="295" dy="0">36</tspan></text><path d="M260 915L330 915" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect x="260" y="908" width="46.666666666666664" height="14" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="295" y="955" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="295" dy="0">6</tspan></text><text x="20" y="1000" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Mean squared error: 18. Mean absolute</tspan><tspan x="20" dy="18">error: 4.</tspan></text><text x="20" y="1110" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Huber at scale one: the nine</tspan><tspan x="20" dy="18">observation has bounded gradient</tspan><tspan x="20" dy="18">minus one at zero. Robust location is</tspan><tspan x="20" dy="18">0.5 here.</tspan></text><text x="20" y="1235" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Treating nine as contamination</tspan><tspan x="20" dy="18">changes the goal if nine is actually</tspan><tspan x="20" dy="18">a genuine tail outcome.</tspan></text><text x="20" y="1360" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A separate decision: suitable</tspan><tspan x="20" dy="18">calibrated probability stays 0.8</tspan></text><text x="20" y="1500" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">False positive cost 9; false negative</tspan><tspan x="20" dy="18">cost one</tspan></text><text x="25" y="1580" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="25" dy="0">Positive: error</tspan><tspan x="25" dy="18">chance 0.2</tspan></text><rect x="25" y="1624" width="29.999999999999993" height="81" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="25" y="1750" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="25" dy="0">Expected cost</tspan><tspan x="25" dy="18">1.8</tspan></text><text x="190" y="1580" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="190" dy="0">Negative: error</tspan><tspan x="190" dy="18">chance 0.8</tspan></text><rect x="190" y="1696" width="120" height="9" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="190" y="1750" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="190" dy="0">Expected cost</tspan><tspan x="190" dy="18">0.8</tspan></text><text x="20" y="1825" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Negative action wins; probability</tspan><tspan x="20" dy="18">threshold 0.9</tspan></text><text x="20" y="1890" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">False positive cost 1; false negative</tspan><tspan x="20" dy="18">cost one</tspan></text><text x="25" y="1970" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="25" dy="0">Positive: error</tspan><tspan x="25" dy="18">chance 0.2</tspan></text><rect x="25" y="2086" width="29.999999999999993" height="9" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="25" y="2140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="25" dy="0">Expected cost</tspan><tspan x="25" dy="18">0.2</tspan></text><text x="190" y="1970" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="190" dy="0">Negative: error</tspan><tspan x="190" dy="18">chance 0.8</tspan></text><rect x="190" y="2086" width="120" height="9" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="190" y="2140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="190" dy="0">Expected cost</tspan><tspan x="190" dy="18">0.8</tspan></text><text x="20" y="2215" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Positive action wins; probability</tspan><tspan x="20" dy="18">threshold 0.5</tspan></text><text x="20" y="2310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Rectangle width is error chance;</tspan><tspan x="20" dy="18">height is error cost, on the same</tspan><tspan x="20" dy="18">scales. Correct decisions cost zero.</tspan></text><text x="20" y="2435" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Changing action costs can change a</tspan><tspan x="20" dy="18">threshold without retraining.</tspan><tspan x="20" dy="18">Validate recall, precision and</tspan><tspan x="20" dy="18">accepted volume; capacity may add</tspan><tspan x="20" dy="18">constraints.</tspan></text><text x="20" y="2580" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Weighted or focal scores need</tspan><tspan x="20" dy="18">probability-meaning and calibration</tspan><tspan x="20" dy="18">checks before using this rule.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> lower bars win within each comparison, not across different losses. Squared distances favor 3; absolute distances favor 0. The separate action example chooses negative despite an unchanged positive probability of 0.8.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="loss-estimation-vs-decision"><strong>Check:</strong> Why can probability 0.8 favor a negative action under these costs?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="loss-estimation-vs-decision">Squared loss selects mean three; absolute loss selects median zero for the same population. Neither target is universally better. Robust location requires a contamination assumption, not automatic deletion of a genuine tail. Separately, a suitable calibrated probability can stay fixed while changed action costs reverse the decision. Validate thresholds, recall and volume. Weighted or focal scores are not automatically calibrated probabilities. <strong>Must the probability model change when only false-positive cost falls to one?</strong></span></figcaption>
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
