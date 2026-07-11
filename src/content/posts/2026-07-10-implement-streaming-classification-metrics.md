---
title: "Implement mergeable streaming classification metrics"
description: "Build a bounded-memory confusion-matrix accumulator with merge, precision, recall, F1, and edge-case handling."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> Implement a streaming binary-classification metric accumulator. It receives batches of labels and predictions, uses bounded memory, supports merging across workers, and reports precision, recall, F1, accuracy, and the confusion matrix.

The important design is to store sufficient statistics rather than every prediction.

## Contract

```python
class BinaryMetrics:
    def update(self, y_true, y_pred): ...
    def merge(self, other): ...
    def compute(self): ...
```

Clarify:

- Are predictions labels or probabilities?
- If probabilities, where is the threshold configured?
- How should undefined precision or recall be represented?
- Are sample weights required?
- Can batches be empty?
- Must updates be thread-safe?

## Reference implementation sketch

```python
class BinaryMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = self.fp = self.tn = self.fn = 0

    def update(self, y_true, y_score):
        if len(y_true) != len(y_score):
            raise ValueError("length mismatch")
        for truth, score in zip(y_true, y_score):
            if truth not in (0, 1):
                raise ValueError("labels must be binary")
            pred = int(score >= self.threshold)
            if truth == 1 and pred == 1:
                self.tp += 1
            elif truth == 0 and pred == 1:
                self.fp += 1
            elif truth == 0 and pred == 0:
                self.tn += 1
            else:
                self.fn += 1

    def merge(self, other):
        if self.threshold != other.threshold:
            raise ValueError("threshold mismatch")
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn
        return self

    @staticmethod
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator else None

    def compute(self):
        precision = self.safe_divide(self.tp, self.tp + self.fp)
        recall = self.safe_divide(self.tp, self.tp + self.fn)
        f1 = None if precision is None or recall is None or precision + recall == 0 \
            else 2 * precision * recall / (precision + recall)
        total = self.tp + self.fp + self.tn + self.fn
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": self.safe_divide(self.tp + self.tn, total),
        }
```

## Why merge works

TP, FP, TN, and FN are additive sufficient statistics for metrics at a fixed threshold. Each worker can accumulate locally, then reducers sum the counts. Storing probabilities would be necessary for metrics across all thresholds such as exact ROC-AUC.

## L4, L5, and L6 signals

- **L4:** correct counts and formulas with basic tests.
- **L5:** explicit undefined-metric policy, threshold contract, merge validation, and weighted or multiclass extension discussion.
- **L6:** distinguishes exact from approximate distributed metrics, addresses delayed labels and slice aggregation, and explains why global F1 cannot be averaged from worker F1 values.

## Tests to write

- Perfect predictions and all-wrong predictions.
- No predicted positives and no actual positives.
- Empty accumulator.
- Batch updates equal one combined update.
- Merging two accumulators equals processing all examples once.
- Threshold mismatch on merge.

## Common mistakes

- Averaging batch-level precision or F1.
- Returning zero for undefined metrics without documenting the policy.
- Storing every prediction despite the streaming requirement.
- Merging workers with different thresholds.
- Confusing macro, micro, and weighted multiclass averages.

## Likely follow-ups

- Extend this to multiclass classification.
- Add sample weights.
- Compute approximate AUC with bounded memory.
- Track metrics by slice without unbounded cardinality.
- Handle labels that arrive days after predictions.

*Related: [precision, recall, and F1](/concepts/precision-recall-f1/), [confusion matrices](/concepts/confusion-matrix-and-classification-metrics/), and [ROC/PR AUC](/concepts/roc-pr-auc/).*
