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

Cross-validation estimates a model's generalization error by repeatedly partitioning the training data into a fitting set and a validation set, training on the first, scoring on the second, and averaging the scores. The right partitioning scheme depends on the data's structure (i.i.d. vs. grouped vs. temporal).

Pick the wrong CV scheme and your validation score is optimistically biased. The model looks great in CV and falls apart in production. The classic failures are (a) k-fold on grouped data leaking the same group into both folds, and (b) random splits on time-series leaking the future into the past.

<!-- visual:cross-validation-preserve-structure -->
<figure class="learning-figure plot-panel" aria-labelledby="cv-structure-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="cv-structure-title">Choose the split that preserves the dependency in your data.</p>
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
	<figcaption><strong>Read it this way:</strong> first ask what a future prediction must generalize beyond. With exchangeable classification examples, preserve the class ratio. With repeated entities, move the whole entity together. With forecasts, train only on the past and validate on the future. The split should imitate the unit that will actually be unseen in production. Original diagram; semantics checked against <a href="https://scikit-learn.org/stable/modules/cross_validation.html">scikit-learn's cross-validation guide</a>.</figcaption>
</figure>

## Standard schemes

### Single hold-out
Split once into train and val (e.g., 80/20). Cheap; high variance in the score.

Use when: large dataset (millions of examples), or when a single CV iteration is too expensive (LLM fine-tuning).

### k-fold
Partition data into $k$ folds. Train $k$ models, each holding out one fold. Average the scores.

- $k = 5$ or $10$ are standard.
- Average and standard deviation across folds give a confidence interval on generalization error.
- Each example is used for training $k-1$ times and validation once.

Use when: i.i.d. data, moderate size, and training is cheap relative to the value of a robust score.

### Stratified k-fold
k-fold where each fold preserves the class distribution of the full dataset. Essential for **imbalanced classification**.

Use when: classification with skewed class frequencies. Always.

### Group / GroupKFold
Each example has a group identifier (user ID, patient ID, document ID). All examples from the same group go to the same fold. Prevents leakage from one group leaking labels into another.

Use when: multiple examples come from the same entity. Examples: user-level recommendation models, patient-level medical models, document-level NLP tasks where one document has many sentences.

### Time-series / TimeSeriesSplit
Folds are chronological. Validation always comes *after* training in time. Earlier folds are smaller; later folds use more history. Never randomize.

Use when: any data with temporal ordering and predictions are forecasts. Examples: demand forecasting, recsys with time-evolving interests, fraud detection.

### Nested CV
Outer loop: estimate generalization. Inner loop: tune hyperparameters within each outer fold.

Use when: hyperparameter tuning matters and you need an unbiased estimate of generalization. Standard in academic ML; rare in industry due to cost.

## When NOT to cross-validate

- **Test set evaluation.** Test set is held out once and scored once at the end. Repeating on test set leaks information.
- **Feature selection on full data.** Selecting features on the entire dataset before CV is leakage. Move feature selection inside the CV loop.
- **Hyperparameter search on full data.** Same. Must be inside the loop or in nested CV.
- **Hidden time leakage.** Even a "random" k-fold on time-stamped data can leak if features include future-derived signals.

## Common pitfalls

- **Random k-fold on time series.** Validation contains points from the same week as training → trivially memorizable. Use chronological splits.
- **Random k-fold on user-grouped data.** Two reviews from the same user end up in different folds; the model learns user-specific patterns and "generalizes" via user identity. Use GroupKFold.
- **Stratifying by the target on regression.** Stratification needs discrete bins; for regression, stratify by quantile bins of the target if needed.
- **Reading too much into one fold's score.** Single-fold scores are noisy; report mean ± std across folds.
- **Tuning on the test set.** Number-one source of fake research results.

## Related

- [Data leakage and point-in-time correctness](/concepts/data-leakage-point-in-time-correctness/). Prevent future, target, group, and preprocessing leakage.
- [A/B testing for ML](/concepts/ab-testing-for-ml/). Online evaluation, complementary to offline CV.
- [Calibration](/concepts/calibration/). Also evaluated on held-out data.
