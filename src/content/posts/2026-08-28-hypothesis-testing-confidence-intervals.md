---
title: "Hypothesis testing and confidence intervals"
description: "Use uncertainty, effect size, and test assumptions to decide whether a measured model gain is likely to be real."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A hypothesis test asks whether observed data is hard to explain under a stated baseline. A confidence interval gives a range of effect sizes that are consistent with the data and the model assumptions.

## Why AI labs care

Model results contain noise. Training seeds, data order, sampled outputs, human ratings, and test examples can all change a score.

A result such as "the new model improved accuracy by 0.4 points" is incomplete. An interviewer will ask:

- How uncertain is the estimate?
- Were the same examples used for both models?
- How many model variants were tried?
- Is the gain large enough to matter?
- Which assumptions make the test valid?

## The basic setup

Suppose a baseline model has performance $m_A$ and a new model has performance $m_B$. The measured change is:

$$
\hat{\delta} = m_B - m_A.
$$

A common null hypothesis is:

$$
H_0: \delta = 0.
$$

The alternative hypothesis says that the true change is not zero, or that it is positive if the direction was chosen before the experiment.

A test statistic compares the measured change with its standard error:

$$
z = \frac{\hat{\delta}}{\operatorname{SE}(\hat{\delta})}.
$$

A large absolute value means the measured change is large relative to the noise estimate.

## What a p-value means

A p-value is the chance of seeing a result at least this extreme if the null hypothesis and test assumptions are true.

It is not:

- the chance that the null hypothesis is true;
- the chance that the result will reproduce;
- the size or value of the improvement;
- proof that the new method caused the change.

A small p-value can describe a tiny effect on a very large dataset. Report the effect size and its interval.

## Confidence intervals

A simple approximate confidence interval is:

$$
\hat{\delta} \pm z^* \operatorname{SE}(\hat{\delta}).
$$

For a 95% interval under a normal approximation, $z^*$ is about 1.96.

Example: a model gain is $0.8$ points with a 95% interval from $0.1$ to $1.5$ points. Zero is outside the interval. The data supports a positive effect under the assumptions. The interval also shows that the true gain may be too small to justify added cost.

**Learning objective:** classify a model-gain confidence interval by whether it rules out no effect and whether it establishes a gain large enough to change the decision.

<!-- visual:model-gain-evidence-decision -->
<figure class="learning-figure plot-panel" aria-labelledby="model-gain-decision-title">
	<p class="visual-kicker">Evidence versus decision</p>
	<p class="visual-title" id="model-gain-decision-title">Compare every interval with zero and the minimum useful gain.</p>
	<svg viewBox="0 0 360 338" role="img" aria-labelledby="model-gain-decision-svg-title model-gain-decision-svg-desc">
		<title id="model-gain-decision-svg-title">Three model-gain confidence intervals compared with zero and a minimum useful gain</title>
		<desc id="model-gain-decision-svg-desc">An original plot of three synthetic 95 percent confidence intervals in percentage points. The first interval runs from minus 0.3 to 1.9 and crosses zero, so the direction is uncertain. The second runs from 0.1 to 1.5; it excludes zero but crosses the predeclared minimum useful gain of 1.0, so there is evidence of a positive effect but practical value remains unresolved. The third runs from 1.2 to 2.0 and lies entirely above the minimum useful gain, supporting a decision-relevant gain under the assumptions.</desc>
		<rect class="viz-plot-bg" x="64" y="54" width="276" height="198" rx="3"></rect>
		<path class="viz-gridline" d="M78 54V252M127 54V252M176 54V252M225 54V252M274 54V252M323 54V252"></path>
		<path class="viz-operating-guide" style="stroke-dasharray:7 5" d="M127 42V263"></path>
		<path style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:2 5" d="M225 42V263"></path>
		<text class="viz-callout" x="127" y="17" text-anchor="middle">no effect</text>
		<text class="viz-callout" x="225" y="17" text-anchor="middle">minimum useful gain</text>
		<text class="viz-label" x="127" y="33" text-anchor="middle">0</text>
		<text class="viz-label" x="225" y="33" text-anchor="middle">+1.0 point</text>
		<text class="viz-axis-label" x="12" y="89">1</text>
		<path class="viz-baseline" d="M98 84H313M98 77V91M313 77V91"></path>
		<circle class="viz-operating-point" cx="205" cy="84" r="4"></circle>
		<text class="viz-label" x="98" y="107" text-anchor="middle">−0.3</text>
		<text class="viz-label" x="313" y="107" text-anchor="middle">+1.9</text>
		<text class="viz-axis-label" x="12" y="139">2</text>
		<path class="viz-axis" d="M137 134H274M137 127V141M274 127V141"></path>
		<circle class="viz-operating-point" cx="205" cy="134" r="4"></circle>
		<text class="viz-label" x="137" y="157" text-anchor="middle">+0.1</text>
		<text class="viz-label" x="274" y="157" text-anchor="middle">+1.5</text>
		<text class="viz-axis-label" x="12" y="189">3</text>
		<path class="viz-axis" style="stroke-width:3" d="M245 184H323M245 177V191M323 177V191"></path>
		<circle class="viz-operating-point" cx="284" cy="184" r="4"></circle>
		<text class="viz-label" x="245" y="207" text-anchor="middle">+1.2</text>
		<text class="viz-label" x="323" y="207" text-anchor="middle">+2.0</text>
		<text class="viz-axis-label" x="12" y="230">VERDICT</text>
		<text class="viz-label" x="78" y="230">1 · Direction uncertain: interval crosses zero.</text>
		<text class="viz-label" x="78" y="247">2 · Positive effect; useful size still uncertain.</text>
		<text class="viz-label" x="78" y="264">3 · Entire interval clears the useful-gain threshold.</text>
		<path class="viz-axis" d="M78 282H323M78 277V287M127 277V287M176 277V287M225 277V287M274 277V287M323 277V287"></path>
		<text class="viz-label" x="78" y="301" text-anchor="middle">−0.5</text>
		<text class="viz-label" x="127" y="301" text-anchor="middle">0</text>
		<text class="viz-label" x="176" y="301" text-anchor="middle">+0.5</text>
		<text class="viz-label" x="225" y="301" text-anchor="middle">+1.0</text>
		<text class="viz-label" x="274" y="301" text-anchor="middle">+1.5</text>
		<text class="viz-label" x="323" y="301" text-anchor="middle">+2.0</text>
		<text class="viz-axis-label" x="200" y="326" text-anchor="middle">model B gain over model A (percentage points)</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> compare each interval's left endpoint with both vertical rules. Crossing zero means the direction remains uncertain. Clearing zero but not the predeclared +1.0-point threshold supports a positive effect without establishing enough gain to act. Only row 3 clears both bars; that conclusion still depends on the test assumptions and whether +1.0 point was chosen before seeing the result. Definitions checked against the <a href="https://doi.org/10.1080/00031305.2016.1154108">ASA statement on p-values</a>, the <a href="https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm">NIST handbook</a>, and <a href="https://doi.org/10.1098/rsta.1937.0005">Neyman's confidence-interval formulation</a>; the graphic and values are original.</figcaption>
</figure>

## Use paired comparisons

Model A and Model B often score the same examples. Their errors are linked. Use the per-example difference instead of treating the two score sets as independent.

For accuracy, define one value per example:

$$
d_i = \mathbf{1}(B\text{ correct}) - \mathbf{1}(A\text{ correct}).
$$

Estimate the mean and uncertainty of $d_i$. A paired bootstrap or a test for paired binary outcomes is usually more efficient than an unpaired test.

For generation tasks, pair outputs by prompt. If humans compare two outputs, randomize their order and keep the prompt as the unit of analysis.

## Type I and Type II errors

- **Type I error:** report a gain when no true gain exists.
- **Type II error:** miss a real gain.
- **Power:** the chance of detecting an effect of a chosen size when it exists.

Low power does not make a negative result useful. Before running an expensive study, choose a minimum effect that would change the decision and estimate the required sample size.

## Multiple comparisons

If a team tries many models, prompts, benchmarks, and slices, one result may look strong by chance.

Good practice:

1. Choose the primary metric before looking at results.
2. Record all tested variants.
3. Treat broad slice analysis as exploration unless it was planned.
4. Confirm promising findings on fresh data.
5. Use a multiple-testing correction when many formal claims are made.

## Common assumptions

Check whether:

- examples are independent, or grouped by user, document, or task;
- the evaluation sample matches the target use;
- the metric has a stable variance;
- the test direction was chosen before seeing the result;
- failed or missing runs were included honestly;
- the test set was not used for model selection.

For repeated prompts from the same user or tasks from the same source, compute uncertainty at the group level.

## In an interview

Use this order:

1. State the decision and primary effect.
2. Name the unit of analysis.
3. Use a paired comparison when both models see the same examples.
4. Report effect size and confidence interval.
5. Discuss power, multiple comparisons, and important slices.
6. State what result would change the launch or research decision.

## Common mistakes

- Reporting only a p-value.
- Calling a non-significant result "no difference."
- Treating prompts from one user as independent samples.
- Picking the best seed and hiding the rest.
- Checking many metrics and reporting only the one that passed.
- Using statistical significance as the launch rule.

*Related: [expectation, variance, covariance, and correlation](/concepts/expectation-variance-covariance-correlation/), [bootstrap and resampling](/concepts/bootstrap-and-resampling/), [A/B testing for ML systems](/concepts/ab-testing-for-ml/), and [design an ablation study](/questions/design-ablation-study/).*