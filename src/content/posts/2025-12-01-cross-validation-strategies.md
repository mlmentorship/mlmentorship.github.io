---
title: "Cross-validation strategies"
description: "Hold-out, k-fold, stratified, grouped, and time-series CV. And when each one is and isn't appropriate."
date: "2025-12-01"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["cross val", "k-fold", "k fold validation", "grouped cross validation", "time series split"]
---

## Summary

Cross-validation scores a training procedure on held-out observations under a declared deployment boundary.
Decide whether the target is a new row, an unseen entity, or a future observation before choosing folds.
The preprocessing, selection, fitting, scoring, and aggregation must all respect that boundary.

For a hypothetical new-patient task, patients A, B, and C each supply three visits.
Hold out each entire patient, fitting on the other two; suppose the respective held-out errors are zero, one, and two.
Equal-patient mean error is $(0+1/3+2/3)/3=1/3$, matching $3/9$ because sizes are equal.
Putting A1 into fitting and A2 into scoring fails this particular unseen-patient claim, even with balanced labels.

<!-- visual:cross-validation-preserve-structure -->
<figure class="learning-figure plot-panel" aria-labelledby="cv-structure-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="cv-structure-title">Choose splits that preserve users, time, and the evaluation claim</p>
	<svg viewBox="0 0 360 440" role="img" aria-labelledby="cv-structure-svg-title cv-structure-svg-desc">
		<title id="cv-structure-svg-title">Three cross-validation splitting constraints</title>
		<desc id="cv-structure-svg-desc">Three panels compare validation rules. For independent classification rows, two folds each contain four negative examples and two positive examples, preserving class proportions. For grouped rows, all examples from groups A and B remain in training while every example from group C is held out for validation. For time-ordered rows, two successive folds use an expanding prefix for training and only later observations for validation. Every region is labelled train or validate, so meaning does not depend on color.</desc>
		<text class="viz-axis-label" x="18" y="23">CHOOSE THE BOUNDARY PRODUCTION WILL ENFORCE</text>
		<rect class="viz-plot-bg" x="18" y="38" width="324" height="108" rx="4"></rect>
		<text class="viz-callout" x="30" y="58">I.I.D. CLASSIFICATION</text>
		<text class="viz-label" x="330" y="58" text-anchor="end">preserve label ratio</text>
		<text class="viz-label" x="30" y="86">fold 1</text>
		<rect class="viz-node viz-node--input" x="78" y="70" width="150" height="25" rx="3"></rect>
		<text class="viz-callout" x="153" y="87" text-anchor="middle">TRAIN  N N P N N P</text>
		<rect class="viz-node viz-node--output" x="236" y="70" width="94" height="25" rx="3"></rect>
		<text class="viz-callout" x="283" y="87" text-anchor="middle">VALIDATE  N N P</text>
		<text class="viz-label" x="30" y="122">fold 2</text>
		<rect class="viz-node viz-node--input" x="78" y="106" width="150" height="25" rx="3"></rect>
		<text class="viz-callout" x="153" y="123" text-anchor="middle">TRAIN  P N N P N N</text>
		<rect class="viz-node viz-node--output" x="236" y="106" width="94" height="25" rx="3"></rect>
		<text class="viz-callout" x="283" y="123" text-anchor="middle">VALIDATE  P N N</text>
		<rect class="viz-plot-bg" x="18" y="158" width="324" height="108" rx="4"></rect>
		<text class="viz-callout" x="30" y="178">REPEATED ENTITIES</text>
		<text class="viz-label" x="330" y="178" text-anchor="end">keep each group intact</text>
		<text class="viz-label" x="30" y="205">TRAIN</text>
		<rect class="viz-node viz-node--input" x="78" y="188" width="72" height="32" rx="3"></rect>
		<text class="viz-callout" x="114" y="208" text-anchor="middle">A1 A2 A3</text>
		<rect class="viz-node viz-node--input" x="158" y="188" width="72" height="32" rx="3"></rect>
		<text class="viz-callout" x="194" y="208" text-anchor="middle">B1 B2 B3</text>
		<text class="viz-label" x="114" y="239" text-anchor="middle">group A</text>
		<text class="viz-label" x="194" y="239" text-anchor="middle">group B</text>
		<text class="viz-label" x="244" y="205">VALIDATE</text>
		<rect class="viz-node viz-node--output" x="244" y="214" width="86" height="32" rx="3"></rect>
		<text class="viz-callout" x="287" y="234" text-anchor="middle">C1 C2 C3</text>
		<text class="viz-label" x="287" y="259" text-anchor="middle">whole group C</text>
		<rect class="viz-plot-bg" x="18" y="278" width="324" height="144" rx="4"></rect>
		<text class="viz-callout" x="30" y="298">ORDERED EVENTS</text>
		<text class="viz-label" x="330" y="298" text-anchor="end">validate only on the future</text>
		<text class="viz-label" x="30" y="330">fold 1</text>
		<rect class="viz-node viz-node--input" x="78" y="313" width="113" height="26" rx="3"></rect>
		<text class="viz-callout" x="134.5" y="331" text-anchor="middle">TRAIN  t1 - t3</text>
		<path class="viz-axis" d="M199 326H215"></path><path class="viz-arrow-forward" d="M215 322L223 326L215 330Z"></path>
		<rect class="viz-node viz-node--output" x="225" y="313" width="105" height="26" rx="3"></rect>
		<text class="viz-callout" x="277.5" y="331" text-anchor="middle">VALIDATE  t4</text>
		<text class="viz-label" x="30" y="371">fold 2</text>
		<rect class="viz-node viz-node--input" x="78" y="354" width="161" height="26" rx="3"></rect>
		<text class="viz-callout" x="158.5" y="372" text-anchor="middle">TRAIN  t1 - t4</text>
		<path class="viz-axis" d="M247 367H263"></path><path class="viz-arrow-forward" d="M263 363L271 367L263 371Z"></path>
		<rect class="viz-node viz-node--output" x="273" y="354" width="57" height="26" rx="3"></rect>
		<text class="viz-callout" x="301.5" y="372" text-anchor="middle">VAL t5</text>
		<text class="viz-label" x="204" y="406" text-anchor="middle">time moves left to right; training history expands →</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> first identify what will be unseen. Stratification can preserve class proportions. Unseen-entity evaluation keeps each entity intact. Forecast evaluation respects information available before its cutoff. These constraints can need composition. The original diagram follows <a href="https://scikit-learn.org/stable/modules/cross_validation.html">scikit-learn's cross-validation guide</a>. <span class="visual-reader-check" data-reader-check="cross-validation-preserve-structure"><strong>Check:</strong> What must stay separated when the same users generate events over time?</span></figcaption>
</figure>

## Standard schemes

### Single hold-out
Fit on one subset and score once on another.
Use when the held-out population is adequate and repeated training costs more than its expected diagnostic value.
One split does not by itself quantify selection or sampling uncertainty.

### k-fold
Fit $k$ models, holding out one fold each time.
For equal-sized folds and the same loss definition, average their scores; unequal folds need estimand-appropriate weights.
For equal-user risk, first average within users instead of letting frequent users silently dominate.
Fold training sets overlap, so their scores are dependent.
The fixture's fold sample standard deviation is $1/3$, descriptive variability rather than a justified confidence interval.

### Stratified k-fold
Preserve label proportions when compatible with the deployment boundary and available independent units.
Check that each fold supports the requested metric.
Class balance does not repair group overlap, future-derived features, or preprocessing leakage.

### Group / GroupKFold
Assign every observation of an entity to the same fold when evaluating unseen entities.
User, patient, or document grouping prevents that identity from crossing the boundary, not every possible leakage mechanism.

### Time-series / TimeSeriesSplit
For forecasting, train on information available before the validation cutoff and score later observations.
Check label availability, overlapping outcome windows, and necessary gaps or purging.
A timestamp alone does not establish that forecasting is the intended evaluation task.

**New task:** Predict later visits for already known patients.

**Picture and change:** Replace unseen-patient exclusion with valid temporal cutoffs while retaining permitted identity history.

**Check:** Training through day ten may score day eleven, not day nine using day-ten information.

**Use when:** Future behavior, rather than unseen identity, is the claim.

### Nested CV
Keep configuration selection inside each outer training subset; reserve its outer validation subset for scoring the selection procedure.
Fit preprocessing, feature selection, and calibration only within their permitted training subsets too.

**New task:** Compare two configurations using two inner folds inside each of three outer folds.

**Picture and change:** Run the inner search, select, then refit on the outer training subset.

**Check:** $3(2)(2)+3=15$ fits, versus three without tuning.
A later full-data refit adds one more.

**Use when:** Selection needs independent outer scoring and the compute budget supports it.
The estimate still reflects its sampling design and training sizes, not an unconditional unbiasedness guarantee.

<!-- learning:core-end -->
<div data-learning-extension>

## When NOT to cross-validate

- **Adapted test set.** Repeatedly changing the procedure after seeing test results removes its untouched-test interpretation.
- **Full-data fitting.** Feature selection, normalization, imputation, and target-derived transformations belong inside their training boundaries.
- **Inappropriate deployment model.** Repeated random splits cannot establish a future-only or unseen-entity claim they do not enforce.

## Common pitfalls

- **Wrong aggregation.** Equal-fold, equal-row, and equal-entity averages need not match when sizes differ.
- **False interval.** Mean plus/minus fold standard deviation is not automatically a confidence interval.
- **Wrong information time.** Event timestamps do not prove that their features and labels were available at training time.

## Reconstruct

<details>
<summary>What does the three-patient result fail to establish?</summary>

It does not provide a causal policy effect, a calibrated confidence interval, or a universal deployment guarantee.
Its interpretation remains conditional on the hypothetical held-out predictions, chosen population, loss, and sampling design.

</details>

</div>

## Related

- [Data leakage and point-in-time correctness](/concepts/data-leakage-point-in-time-correctness/). Prevent future, target, group, and preprocessing leakage.
- [A/B testing for ML](/concepts/ab-testing-for-ml/). Online evaluation, complementary to offline CV.
- [Calibration](/concepts/calibration/). Also evaluated on held-out data.
