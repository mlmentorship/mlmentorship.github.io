---
title: "How do you deal with class imbalance in 2026?"
description: "Class weighting and SMOTE are the textbook answers and often the wrong ones. The senior answer matches the technique to the imbalance ratio, the cost asymmetry, and the metric you actually care about."
date: "2026-05-07"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: ML breadth, especially in fraud, medical, search, and rare-event domains.*

The L4 candidate names SMOTE. The L6 candidate asks what the cost asymmetry is, what metric is being optimized, and whether the imbalance is even the problem.

## What an L4 answer sounds like

> "I'd oversample the minority class, undersample the majority, or use SMOTE to generate synthetic minority examples. Class weights in the loss function also help."

These are tools. They're often the wrong ones. You've consumed a stack-overflow checklist.

## What an L5 answer sounds like

> "First, I'd ask what's actually being measured. 'Class imbalance is bad' is only true if your metric is sensitive to it. Accuracy on a 99:1 dataset is misleading; AUC, average precision, or F1 are not. Many imbalance fixes just paper over a metric problem.
>
> If imbalance does need addressing, I'd consider:
>
> 1. **Threshold tuning**. Train a calibrated model, pick a decision threshold that matches the cost trade-off. Often the right answer.
> 2. **Class weighting in the loss**. Weight rare-class examples more. Cheap, works reasonably well.
> 3. **Focal loss**. Down-weight easy examples (most majority-class), focus gradient on hard examples (typically minority-class boundary cases).
> 4. **Undersampling the majority** if the majority is huge and noisy. Makes training cheaper and often improves quality.
> 5. **Oversampling / SMOTE** if the minority is tiny. SMOTE works for tabular but is fragile for high-dim data (images, text); GAN/VAE-based augmentation is rarely worth the complexity.
> 6. **Two-stage models**: a high-recall first stage, then a high-precision filter. Common in fraud detection."

This is L5. You've named the metric problem first, then sequenced the techniques by typical effectiveness.

## What an L6 answer sounds like

> "...practical points:
>
> **Calibration breaks under sampling-based fixes.** If you oversample the minority class, the predicted probabilities no longer correspond to true class probabilities. Either correct them post-hoc (Platt scaling, isotonic regression) or skip the resampling and use class weighting.
>
> **Imbalance often hides a label problem.** A '99:1' fraud dataset usually has many false negatives in the majority class (frauds you didn't catch). Imbalance techniques applied to noisy labels just amplify the noise. Worth investigating label quality before applying any fix.
>
> **For deep nets, hard-example mining is more useful than class balancing.** OHEM (online hard example mining), focal loss, or a learned curriculum often dominates static class weights.
>
> **For LLMs and large pretraining, imbalance is rarely an issue.** The model sees enough examples that even rare patterns are well-represented in absolute terms."

## Tells that get you a strong-hire vote

- You **question whether imbalance is the problem** before applying fixes.
- You bring up **threshold tuning** as the first response, not a model change.
- You distinguish **calibration impact** of resampling vs class weighting.
- You name **focal loss** for hard-example focus.
- You consider **two-stage architectures** for high-imbalance settings.

## Tells that get you down-leveled

- Reaching for SMOTE as the default.
- Reporting accuracy on imbalanced data.
- No mention of calibration impact.
- Treating "balance the classes" as a goal independent of the metric.

## Common follow-up

"What metric would you use for a 99:1 fraud problem?"

The L6 answer:

> "Depends on the cost structure. If false negatives (missed fraud) cost much more than false positives (legitimate transaction declined), I'd track recall at a fixed precision (or precision at a fixed recall, whichever the business commits to). I'd report the full precision-recall curve, not the average precision alone, because business decisions are made at specific operating points. AUROC is misleading at high imbalance; average precision (AUPRC) is more honest."

---

*Related: [Calibration](/reference/calibration/), [How to choose a loss function](/interviews/how-to-choose-loss-function/), [Walk me through bias-variance tradeoff](/interviews/bias-variance-tradeoff/).*
