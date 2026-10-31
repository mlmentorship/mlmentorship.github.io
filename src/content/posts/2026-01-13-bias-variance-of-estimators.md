---
title: "Bias and variance of estimators"
description: "An estimator has bias (systematic error) and variance (sample-to-sample wobble). Mean-squared error decomposes into the two."
date: "2026-01-13"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

For an estimator $\hat\theta$ of a parameter $\theta$, **bias** is $\mathbb{E}[\hat\theta] - \theta$ and **variance** is $\mathbb{E}[(\hat\theta - \mathbb{E}[\hat\theta])^2]$. Mean-squared error decomposes as $\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Variance}$.

This decomposition is the statistical version of the [bias–variance tradeoff](/questions/bias-variance-tradeoff/) familiar from ML: complex models have low bias but high variance; simple models have high bias but low variance. The same accounting applies to any estimator. Sample mean, regularized regression coefficient, importance sampling weight.

## The decomposition

For estimator $\hat\theta$ of a fixed (non-random) parameter $\theta$:

$$
\begin{aligned}
\mathrm{MSE}(\hat\theta) &= \mathbb{E}[(\hat\theta - \theta)^2] \\
&= \mathbb{E}[(\hat\theta - \mathbb{E}\hat\theta)^2] + (\mathbb{E}\hat\theta - \theta)^2 \\
&= \mathrm{Var}(\hat\theta) + \mathrm{Bias}(\hat\theta)^2.
\end{aligned}
$$

The cross-term vanishes because $\mathbb{E}[\hat\theta - \mathbb{E}\hat\theta] = 0$. Two-line derivation; central to all of statistics.

<!-- visual:estimator-error-reference-points -->
<figure class="learning-figure" aria-labelledby="estimator-error-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="estimator-error-title">Spread is measured from the mean, bias from the truth</p><div class="visual-argument" data-visual-argument="estimator-error-reference-points" data-argument-kind="geometry" data-argument-composition="geometric-action"><svg viewBox="0 0 360 501" role="img" aria-label="Bias and spread use different reference points. This illustrative estimator has four equally likely outcomes. Its mean is (3,0), one unit from truth (2,0). Squared spread around its mean is 2. Mean squared error is 3."><title>Bias and spread use different reference points</title><desc>This illustrative estimator has four equally likely outcomes. Its mean is (3,0), one unit from truth (2,0). Squared spread around its mean is 2. Mean squared error is 3.</desc><g opacity=".22"><path d="M34 330L34 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 330L322 330" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M82 330L82 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 282L322 282" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M130 330L130 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 234L322 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M178 330L178 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 186L322 186" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M226 330L226 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 138L322 138" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M274 330L274 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 90L322 90" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M322 330L322 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M34 42L322 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path></g><path d="M34 282L322 282" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M82 330L82 42" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M226 282L178 330" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="178" cy="330" r="6" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><path d="M226 282L178 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="178" cy="234" r="6" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><path d="M226 282L274 330" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="274" cy="330" r="6" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><path d="M226 282L274 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="274" cy="234" r="6" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><path d="M178 282L226 282M218 286L226 282L218 278" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="178" cy="282" r="8" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="226" cy="282" r="8" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Repeat the estimator, not the truth</tspan></text><text x="20" y="91" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Gray points: four possible estimates</tspan></text><text x="20" y="378" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Truth to mean: bias</tspan></text><text x="20" y="420" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Mean to outcomes: spread</tspan></text><text x="20" y="465" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Mean squared error: 1 + 2 = 3</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 250" role="img" aria-labelledby="estimator-error-svg-title estimator-error-svg-desc">
			<title id="estimator-error-svg-title">Bias and variance use different reference points</title>
			<desc id="estimator-error-svg-desc">A sampling distribution contains seven estimates centered on the estimator expectation. A solid diamond marks the true parameter to the left, while a dashed vertical guide and circle mark the estimator expectation. Variance is labelled as squared spread of estimates around the expectation. Bias is labelled as the distance from the expectation to the true parameter. The mean squared error equals variance plus squared bias.</desc>
			<path d="M102 120 C126 118 139 96 151 73 C164 47 184 35 207 35 C230 35 250 47 263 73 C275 96 288 118 318 120 Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
			<text class="viz-callout" x="210" y="22" text-anchor="middle">Sampling distribution of &#952;&#770;</text>
			<circle cx="142" cy="120" r="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
			<circle cx="171" cy="120" r="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
			<circle cx="194" cy="120" r="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
			<circle cx="210" cy="120" r="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
			<circle cx="229" cy="120" r="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
			<circle cx="253" cy="120" r="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
			<circle cx="284" cy="120" r="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
			<path class="viz-axis" d="M28 145H332"></path>
			<path d="M112 66V156" style="fill:none;stroke:var(--c-text);stroke-width:2"></path>
			<path d="M107 145L112 140L117 145L112 150Z" style="fill:var(--viz-neutral-bg);stroke:var(--c-text);stroke-width:2"></path>
			<text class="viz-axis-label" x="112" y="174" text-anchor="middle">True &#952;</text>
			<path d="M210 34V156" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:2;stroke-dasharray:5 4"></path>
			<circle cx="210" cy="145" r="5" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></circle>
			<text class="viz-axis-label" x="210" y="174" text-anchor="middle">Center E[&#952;&#770;]</text>
			<path d="M142 91H205M215 91H284" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.7;stroke-dasharray:3 3"></path>
			<path d="M142 87V95M205 87V95M215 87V95M284 87V95" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.7"></path>
			<text class="viz-axis-label" x="210" y="78" text-anchor="middle">Variance: squared spread around E[&#952;&#770;]</text>
			<path d="M112 194H210M112 188V200M210 188V200" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
			<text class="viz-axis-label" x="161" y="215" text-anchor="middle">Bias = E[&#952;&#770;] - &#952;</text>
			<text class="viz-callout" x="180" y="240" text-anchor="middle">MSE = variance + bias&#178;</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> Re-sample the data and the estimates wobble around their own center, E[&#952;&#770;]; that squared wobble is variance. The center itself can miss the true &#952;; that displacement is bias. Averaging squared distance all the way to &#952; gives variance plus bias squared.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="estimator-error-reference-points"><strong>Check:</strong> Around which point is variance measured, and around which point is total estimation error measured?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="estimator-error-reference-points">This illustrative estimator has four equally likely outcomes. Its mean is (3,0), one unit from truth (2,0). Squared spread around its mean is 2. Mean squared error is 3. <br><strong>If all four estimates move together, does their spread around their own mean change?</strong></span></figcaption>
</figure>

## Why biased estimators can be useful

Unbiased estimators ($\mathbb{E}\hat\theta = \theta$) are not always optimal. A biased estimator with much lower variance can have lower MSE.

Examples:

| Estimator | Bias | Variance | When better |
|----------|------|----------|-------------|
| Sample mean | 0 | $\sigma^2/n$ | universal |
| Sample variance with $n-1$ | 0 | larger | unbiased baseline |
| Sample variance with $n$ (MLE) | small negative | smaller | when minimizing MSE |
| Ridge regression | nonzero | smaller than OLS | when $X^\top X$ is ill-conditioned |
| Stein estimator | shrinkage bias | strictly lower | always for $\ge 3$ dimensions |

The James-Stein estimator (1961) famously dominates the sample mean in $\ge 3$ dimensions despite being biased.

## Connection to ML model selection

In supervised learning, the same decomposition holds for the prediction error of a model:

$$
\mathbb{E}[(y - \hat f(x))^2] = \mathrm{Var}(\hat f(x)) + \mathrm{Bias}(\hat f(x))^2 + \sigma_\varepsilon^2.
$$

The $\sigma_\varepsilon^2$ term is irreducible noise. Increasing model capacity decreases bias but increases variance; regularization shifts the tradeoff toward higher bias.

## Common pitfalls

- **Equating "biased" with "bad."** Many useful estimators are biased; lower MSE is what matters.
- **Reporting variance without specifying what's random.** "Variance of the estimator" is over re-sampling the data; "variance of the prediction" is over both data and inputs. Different objects.
- **Forgetting that the cross-term vanishes only against the *expectation* of $\hat\theta$.** Random fixed offsets ruin the decomposition.
- **Confusing estimator variance with model variance.** Estimator variance is a property of the estimation procedure; model variance is a property of the model class.
