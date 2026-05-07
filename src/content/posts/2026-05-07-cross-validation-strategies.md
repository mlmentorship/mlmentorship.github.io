---
title: "Cross-validation strategies"
description: "Hold-out, k-fold, stratified, grouped, and time-series CV. And when each one is and isn't appropriate."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Cross-validation estimates a model's generalization error by repeatedly partitioning the training data into a fitting set and a validation set, training on the first, scoring on the second, and averaging the scores. The right partitioning scheme depends on the data's structure (i.i.d. vs. grouped vs. temporal).

## Why it matters

Pick the wrong CV scheme and your validation score is optimistically biased. The model looks great in CV and falls apart in production. The classic failures are (a) k-fold on grouped data leaking the same group into both folds, and (b) random splits on time-series leaking the future into the past.

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

- [A/B testing for ML](/reference/ab-testing-for-ml/). Online evaluation, complementary to offline CV.
- [Calibration](/reference/calibration/). Also evaluated on held-out data.
