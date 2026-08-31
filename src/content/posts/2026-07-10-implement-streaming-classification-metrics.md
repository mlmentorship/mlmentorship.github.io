---
title: "Implement mergeable streaming classification metrics"
description: "Build a bounded-memory confusion-matrix accumulator with merge, precision, recall, F1, and edge-case handling."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> Implement a streaming binary-classification metric accumulator. It receives batches of labels and predictions, uses bounded memory, supports merging across workers, and reports precision, recall, F1, accuracy, and the confusion matrix.

This is distributed model evaluation in miniature: score a stream too large to hold in memory, then merge per-worker accumulators into one global result. The design hinges on storing sufficient statistics (the four confusion-matrix counts) rather than every prediction, because those counts are exactly what makes the metric mergeable across shards.

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

<!-- visual:streaming-metrics-counts-before-f1 -->
<figure class="learning-figure" aria-labelledby="streaming-counts-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="streaming-counts-title">What should workers merge: confusion counts or finished F1 scores?</p>
	<div class="visual-grid--two" role="group" aria-label="Two shards merge confusion counts before computing one global F1 score">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 245" role="img" aria-labelledby="count-merge-title count-merge-desc">
				<title id="count-merge-title">Confusion-count states add component by component</title>
				<desc id="count-merge-desc">Shard A stores true positives 1, false positives 0, true negatives 9, and false negatives 0. Shard B stores true positives 1, false positives 9, true negatives 1, and false negatives 9. Componentwise addition produces the global state: true positives 2, false positives 9, true negatives 10, and false negatives 9. Only four counts are retained regardless of stream size.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="212" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">1 · REDUCE THE ADDITIVE STATE</text>
				<rect class="viz-node viz-node--input" x="20" y="39" width="260" height="45" rx="4"></rect>
				<text class="viz-node-label" x="56" y="58">SHARD A</text>
				<text class="viz-label" x="150" y="74" text-anchor="middle">TP 1 · FP 0 · TN 9 · FN 0</text>
				<text class="viz-callout" x="150" y="104" text-anchor="middle">+</text>
				<rect class="viz-node viz-node--input" x="20" y="115" width="260" height="45" rx="4"></rect>
				<text class="viz-node-label" x="56" y="134">SHARD B</text>
				<text class="viz-label" x="150" y="150" text-anchor="middle">TP 1 · FP 9 · TN 1 · FN 9</text>
				<text class="viz-callout" x="150" y="180" text-anchor="middle">= · add matching fields</text>
				<rect class="viz-node viz-node--output" x="20" y="190" width="260" height="39" rx="4"></rect>
				<text class="viz-node-label" x="65" y="214">GLOBAL</text>
				<text class="viz-label" x="189" y="214" text-anchor="middle">TP 2 · FP 9 · TN 10 · FN 9</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 245" role="img" aria-labelledby="f1-after-merge-title f1-after-merge-desc">
				<title id="f1-after-merge-title">Averaging shard F1 differs from F1 computed from global counts</title>
				<desc id="f1-after-merge-desc">The wrong path computes shard A F1 as 1.00 and shard B F1 as 0.10, then averages them to 0.55. The correct path first uses the merged counts and computes F1 as two times 2 divided by two times 2 plus 9 plus 9, which is 4 over 22 or approximately 0.18. F1 is nonlinear, so 0.55 and 0.18 differ.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="212" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">2 · COMPUTE THE RATIO ONCE</text>
				<rect x="20" y="39" width="260" height="72" rx="5" style="fill:var(--c-surface);stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:5 3"></rect>
				<text class="viz-axis-label" x="30" y="57">WRONG · AVERAGE WORKER F1</text>
				<text class="viz-label" x="150" y="79" text-anchor="middle">(1.00 + 0.10) / 2 = 0.55</text>
				<path d="M55 67L245 97M245 67L55 97" style="fill:none;stroke:var(--viz-edge);stroke-width:1.3"></path>
				<rect class="viz-node viz-node--focus" x="20" y="129" width="260" height="91" rx="5"></rect>
				<text class="viz-axis-label" x="30" y="148">CORRECT · F1 FROM GLOBAL COUNTS</text>
				<text class="viz-node-label" x="150" y="173">F1 = 2TP / (2TP + FP + FN)</text>
				<text class="viz-label" x="150" y="194" text-anchor="middle">= 4 / (4 + 9 + 9) = 4 / 22</text>
				<text class="viz-callout" x="150" y="213" text-anchor="middle">approximately 0.18 · not 0.55</text>
				<text class="viz-label" x="150" y="233" text-anchor="middle">F1 is nonlinear; its inputs are mergeable.</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> retain the four fixed-threshold counts on every worker, add matching fields, and compute F1 once from the totals. Computing early loses each shard's denominator: here the mean worker F1 is <code>0.55</code>, but the same records have global F1 <code>4/22 ≈ 0.18</code>. The count state is mergeable; the finished ratio is not. Original example checked against the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html">scikit-learn confusion-matrix</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html">F1 definitions</a> plus <a href="https://lightning.ai/docs/torchmetrics/stable/pages/implement.html">TorchMetrics state-reduction guidance</a>.</figcaption>
</figure>

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

## Common follow-ups

- Extend this to multiclass classification.
- Add sample weights.
- Compute approximate AUC with bounded memory.
- Track metrics by slice without unbounded cardinality.
- Handle labels that arrive days after predictions.

*Related: [precision, recall, and F1](/concepts/precision-recall-f1/), [confusion matrices](/concepts/confusion-matrix-and-classification-metrics/), and [ROC/PR AUC](/concepts/roc-pr-auc/).*
