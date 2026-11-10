---
title: "Hypothesis testing and confidence intervals"
description: "Quantify paired model gains under sampling assumptions, then compare uncertainty with a predeclared useful-effect threshold."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A hypothesis test compares observed data with a null model under explicit sampling and test assumptions.
A confidence interval quantifies uncertainty using a procedure whose coverage is defined across repeated samples.
Statistical evidence of improvement is different from evidence of a practically useful gain.

## The basic setup

Compare two fixed models on independent paired examples from a declared target population.
Before inspection, choose the accuracy difference, test direction, significance level, and minimum useful gain of one percentage point.
All counts and intervals here are teaching fixtures, not measured model results.

For scores $m_A,m_B$, the observed change is

$$
\hat{\delta} = m_B - m_A.
$$

A common null hypothesis is:

$$
H_0: \delta = 0.
$$

Choose a two-sided alternative $\delta\ne0$, or a predeclared directional alternative.
A standardized statistic compares the estimate with its standard error:

$$
z = \frac{\hat{\delta}}{\operatorname{SE}(\hat{\delta})}.
$$

Its null distribution and standard error require justification; the formula alone is not a valid test.

## What a p-value means

A p-value is a null-conditional probability of a statistic at least as extreme as observed.
It is not the null's posterior probability, replication probability, effect size, or evidence of causation by itself.

## Confidence intervals

A normal-approximation interval uses

$$
\hat{\delta} \pm z^* \operatorname{SE}(\hat{\delta}).
$$

For approximate 95% coverage, use $z^*=1.96$ under the matching assumptions.
Across repetitions, about 95% of intervals constructed this way cover the fixed effect when the approximation is accurate.
This is not a 95% posterior probability statement about the realized interval.

<!-- visual:model-gain-evidence-decision -->
<figure class="learning-figure plot-panel" aria-labelledby="model-gain-decision-title">
	<p class="visual-kicker">Evidence versus decision</p>
	<p class="visual-title" id="model-gain-decision-title">Use paired uncertainty before claiming direction or a worthwhile model gain</p><div class="visual-argument" data-visual-argument="model-gain-evidence-decision" data-argument-kind="bars" data-argument-composition="gain-floor"><svg viewBox="0 0 360 2400" role="img" aria-label="Use paired uncertainty before claiming direction or a worthwhile model gain. Keep one difference per paired example. The first interval spans zero and the useful-gain threshold, so direction remains uncertain. More new independent pairs narrow it above zero, but it still crosses the threshold. Positive evidence is not proof of a worthwhile gain. Duplicating observations supplies no new independent information. Coverage is a repeated-sampling property, not a posterior probability. Predeclare the test and claim family; twenty claims need multiplicity control."><title>Use paired uncertainty before claiming direction or a worthwhile model gain</title><desc>Keep one difference per paired example. The first interval spans zero and the useful-gain threshold, so direction remains uncertain. More new independent pairs narrow it above zero, but it still crosses the threshold. Positive evidence is not proof of a worthwhile gain. Duplicating observations supplies no new independent information. Coverage is a repeated-sampling property, not a posterior probability. Predeclare the test and claim family; twenty claims need multiplicity control.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Paired information narrows</tspan><tspan x="20" dy="18">uncertainty, not the decision rule</tspan></text><text x="20" y="112" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Two fixed models; independent paired</tspan><tspan x="20" dy="18">examples from a declared population.</tspan><tspan x="20" dy="18">Predeclare test direction, level and</tspan><tspan x="20" dy="18">useful gain: one percentage point.</tspan></text><text x="20" y="330" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">1000 pairs: 20 B-only wins, 12 A-only</tspan><tspan x="20" dy="18">wins</tspan></text><text x="20" y="415" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">968 zero differences; estimate 0.8</tspan><tspan x="20" dy="18">point</tspan></text><text x="20" y="506" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Paired variance 0.031968; standard</tspan><tspan x="20" dy="18">error 0.5654 point</tspan></text><path d="M45 642L315 642" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M99 595L99 680" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M207 580L207 695" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M65.71565632972798 642L305.08434367027206 642" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M65.71565632972798 629L65.71565632972798 655" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M305.08434367027206 629L305.08434367027206 655" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle cx="185.4" cy="642" r="7" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="99" y="741" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="99" dy="0">0</tspan></text><text x="207" y="741" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="207" dy="0">1</tspan></text><text x="315" y="741" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="315" dy="0">2</tspan></text><text x="20" y="829" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Approximate 95-percent interval:</tspan><tspan x="20" dy="18">-0.308 to 1.908 points</tspan></text><text x="20" y="970" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">4000 pairs: 80 B-only wins, 48 A-only</tspan><tspan x="20" dy="18">wins</tspan></text><text x="20" y="1055" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">3872 zero differences; estimate 0.8</tspan><tspan x="20" dy="18">point</tspan></text><text x="20" y="1146" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Paired variance 0.031944; standard</tspan><tspan x="20" dy="18">error 0.2826 point</tspan></text><path d="M45 1282L315 1282" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M99 1235L99 1320" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M207 1220L207 1335" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M125.5802788022458 1282L245.2197211977542 1282" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M125.5802788022458 1269L125.5802788022458 1295" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M245.2197211977542 1269L245.2197211977542 1295" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle cx="185.4" cy="1282" r="7" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="99" y="1381" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="99" dy="0">0</tspan></text><text x="207" y="1381" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="207" dy="0">1</tspan></text><text x="315" y="1381" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="315" dy="0">2</tspan></text><text x="20" y="1469" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Approximate 95-percent interval:</tspan><tspan x="20" dy="18">0.246 to 1.354 points</tspan></text><text x="20" y="1660" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same ruler: grey zero; dashed red</tspan><tspan x="20" dy="18">one-point useful threshold. First</tspan><tspan x="20" dy="18">interval spans zero and the</tspan><tspan x="20" dy="18">threshold. Second is positive but</tspan><tspan x="20" dy="18">still crosses the useful threshold.</tspan></text><text x="20" y="1820" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Four thousand new independent pairs</tspan><tspan x="20" dy="18">supply more information. Repeating</tspan><tspan x="20" dy="18">the original records four times does</tspan><tspan x="20" dy="18">not. Keep discordant pairs; separate</tspan><tspan x="20" dy="18">model marginals lose covariance.</tspan></text><text x="20" y="1970" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">First interval also fails to</tspan><tspan x="20" dy="18">establish equivalence within a</tspan><tspan x="20" dy="18">half-point margin. Sampling</tspan><tspan x="20" dy="18">assumptions and normal-approximation</tspan><tspan x="20" dy="18">coverage need justification.</tspan></text><text x="20" y="2115" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Coverage belongs to the</tspan><tspan x="20" dy="18">repeated-sampling procedure, not a</tspan><tspan x="20" dy="18">95-percent posterior claim about this</tspan><tspan x="20" dy="18">interval. A p-value is</tspan><tspan x="20" dy="18">null-conditional, not the chance the</tspan><tspan x="20" dy="18">null is true.</tspan></text><text x="20" y="2270" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Twenty prespecified formal claims at</tspan><tspan x="20" dy="18">family-wise level 0.05: Bonferroni</tspan><tspan x="20" dy="18">threshold 0.0025 each. A p-value 0.01</tspan><tspan x="20" dy="18">does not reject under that procedure.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 338" role="img" aria-labelledby="model-gain-decision-svg-title model-gain-decision-svg-desc">
		<title id="model-gain-decision-svg-title">Three model-gain confidence intervals compared with zero and a minimum useful gain</title>
		<desc id="model-gain-decision-svg-desc">An original plot of three synthetic 95 percent confidence intervals in percentage points. The first interval runs from minus 0.3 to 1.9 and crosses zero, so the direction is uncertain. The second runs from 0.1 to 1.5; it excludes zero but crosses the predeclared minimum useful gain of 1.0, so there is evidence of a positive effect but practical value remains unresolved. The third runs from 1.2 to 2.0 and lies entirely above the minimum useful gain, supporting a decision-relevant gain under the assumptions.</desc>
		<rect class="viz-plot-bg" x="64" y="54" width="276" height="198" rx="3"></rect>
		<path class="viz-gridline" d="M78 54V252M127 54V252M176 54V252M225 54V252M274 54V252M323 54V252"></path>
		<path class="viz-operating-guide" style="stroke-dasharray:7 5" d="M127 42V218"></path>
		<path style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:2 5" d="M225 42V218"></path>
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
	<div class="visual-derivation-caption">compare each interval with zero and the predeclared useful gain. Row 1 establishes neither direction nor usefulness. Row 2 excludes zero but still crosses one point; row 3 lies wholly above one point. The paired calculation below produces row 1's rounded endpoints; the other rows are separate synthetic examples. Definitions retain the <a href="https://doi.org/10.1080/00031305.2016.1154108">ASA</a>, <a href="https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm">NIST</a>, and <a href="https://doi.org/10.1098/rsta.1937.0005">Neyman</a> reference provenance.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="model-gain-evidence-decision"><strong>Check:</strong> Why does interval 2 support a positive effect without establishing that the gain exceeds one percentage point?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="model-gain-evidence-decision">Keep one difference per paired example. The first interval spans zero and the useful-gain threshold, so direction remains uncertain. More new independent pairs narrow it above zero, but it still crosses the threshold. Positive evidence is not proof of a worthwhile gain. Duplicating observations supplies no new independent information. Coverage is a repeated-sampling property, not a posterior probability. Predeclare the test and claim family; twenty claims need multiplicity control. <strong>Does the positive second interval establish a gain exceeding one point?</strong></span></figcaption>
</figure>

## Use paired comparisons

Keep one difference per paired example:

$$
d_i = \mathbf{1}(B\text{ correct}) - \mathbf{1}(A\text{ correct}).
$$

Among 1,000 pairs, B alone is correct on 20 and A alone on 12; 968 differences are zero.
Therefore $\bar d=(20-12)/1000=0.008$, and $\sum_i d_i^2=32$.
The paired sample variance and standard error are

$$
s_d^2=\frac{\sum_i d_i^2-n\bar d^2}{n-1}
=\frac{32-1000(0.008)^2}{999}\approx0.031967968,
\qquad \operatorname{SE}=\sqrt{s_d^2/n}\approx0.005654022.
$$

The interval is $0.008\pm1.96(0.005654022)$, or approximately $[-0.3082,1.9082]$ percentage points.
The statistic is about $1.415$.
Discarding the 32 discordant pairs loses the covariance needed for this variance.
The interval neither establishes direction nor equivalence within a $\pm0.5$-point margin.

### Increase independent information

**New task:** evaluate 4,000 new independent pairs with the same hypothetical discordance proportions.

**Picture and change:** use 80 positive differences, 48 negative differences, and 3,872 zeros.
Recompute variance and standard error rather than copying the old interval.

**Check:** $\bar d=0.008$, $s_d^2=(128-4000(0.008)^2)/3999$, and SE is approximately $0.002826$.
The interval becomes approximately $[0.246,1.354]$ points: positive, but still crossing the useful threshold.
Duplicating the original observations four times would not provide this independent information.

**Use when:** fresh evaluation units can improve precision without changing the estimand.

### Make twenty formal claims

**New task:** control family-wise error at $0.05$ for twenty prespecified tests.

**Picture and change:** Bonferroni compares each p-value with $0.05/20$.

**Check:** the threshold is $0.0025$; p-value $0.01$ does not reject under this procedure.
Record the entire claim family rather than selecting its size after seeing results.

**Use when:** formal claims span multiple metrics, variants, or slices.

<!-- learning:core-end -->

<div data-learning-extension>

## Why AI labs care

Evaluation examples, training seeds, generated outputs, and human ratings are different randomness sources.
The paired fixture addresses evaluation-example uncertainty conditional on fixed models, not every training recipe's variation.
Report failed and missing runs; do not select the best seed and hide the rest.

## Type I and Type II errors

- **Type I error:** report a gain when no true gain exists.
- **Type II error:** miss a real gain.
- **Power:** the chance of detecting an effect of a chosen size when it exists.

Non-rejection does not establish equality.
An equivalence claim needs a predeclared margin and an appropriate procedure.
Choose a minimum decision-relevant effect and plan power under plausible variance and clustering before an expensive study.

## Multiple comparisons

Model or prompt selection on the evaluation set changes what the reported interval covers.
Keep exploratory slices distinct from planned claims and confirm selected findings on fresh data.
Repeated peeking needs a sequentially valid design, not repeated use of a fixed-sample threshold.

## Common assumptions

Pair generation outputs by prompt and randomize human presentation order.
If users or documents contribute repeated observations, retain clusters when estimating uncertainty.
Normal approximations need adequate sampling behavior; paired bootstrap and paired-binary tests require their own assumptions.
Representative sampling, causal assignment, and training-run uncertainty are separate concerns.

## In an interview

An interval is $[0.1,1.5]$ points and the useful threshold is one point. What follows?

<details><summary>Answer</summary>

It supports a positive effect under the matching test and assumptions, but not a gain exceeding one point.
It is not a posterior probability statement or a complete launch decision.

</details>

## Common mistakes

CI exclusion corresponds to rejection only for a matching direction, level, and test construction.
A statistically precise tiny gain can still lose against cost, latency, or safety constraints.

</div>

*Related: [expectation, variance, covariance, and correlation](/concepts/expectation-variance-covariance-correlation/), [bootstrap and resampling](/concepts/bootstrap-and-resampling/), [A/B testing for ML systems](/concepts/ab-testing-for-ml/), and [design an ablation study](/questions/design-ablation-study/).*