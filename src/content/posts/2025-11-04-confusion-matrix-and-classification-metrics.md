---
title: "Confusion matrix and classification metrics"
description: "The 2x2 (or KxK) table of predictions vs. truth that every classification metric is computed from. The Rosetta stone of binary classification."
date: "2025-11-04"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **confusion matrix** is a $K \times K$ table where rows are true classes and columns are predicted classes (or vice versa, depending on convention). Each cell counts how many examples had that (true, predicted) pair. Every standard classification metric. Accuracy, precision, recall, specificity, F1, balanced accuracy. Is a function of the confusion-matrix counts.

Reporting a single classification metric is almost always insufficient. The confusion matrix:

- Shows **which** errors the model makes (e.g., "confuses cat with dog 30% of the time").
- Reveals class imbalance issues that aggregate metrics hide.
- Is the source of truth for choosing decision thresholds.

A senior ML practitioner should always look at the confusion matrix before reporting a metric.

## The binary confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN (True Neg) | FP (False Pos, Type I) |
| **Actual 1** | FN (False Neg, Type II) | TP (True Pos) |

<!-- visual:confusion-matrix-denominators -->
<figure class="learning-figure" aria-labelledby="cm-denominators-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="cm-denominators-title">Precision reads a prediction column. Recall reads an actual-class row.</p>
	<div class="visual-grid--two">
		<section class="visual-panel" aria-labelledby="precision-panel-title">
			<h4 id="precision-panel-title">Precision: trust a positive prediction</h4>
			<p>Highlight the predicted-positive column.</p>
			<table class="cm-grid" aria-label="Confusion matrix with the predicted-positive column highlighted for precision">
				<thead><tr><th>Actual</th><th>Pred 0</th><th>Pred 1</th></tr></thead>
				<tbody>
					<tr><th scope="row">0</th><td><strong>900</strong> TN</td><td class="cm-selected"><strong>90</strong> FP</td></tr>
					<tr><th scope="row">1</th><td><strong>2</strong> FN</td><td class="cm-selected"><strong>8</strong> TP</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">precision = 8 / (8 + 90) = 8%</p>
		</section>
		<section class="visual-panel" aria-labelledby="recall-panel-title">
			<h4 id="recall-panel-title">Recall: find the actual positives</h4>
			<p>Highlight the actual-positive row.</p>
			<table class="cm-grid" aria-label="Confusion matrix with the actual-positive row highlighted for recall">
				<thead><tr><th>Actual</th><th>Pred 0</th><th>Pred 1</th></tr></thead>
				<tbody>
					<tr><th scope="row">0</th><td><strong>900</strong> TN</td><td><strong>90</strong> FP</td></tr>
					<tr><th class="cm-selected" scope="row">1</th><td class="cm-selected"><strong>2</strong> FN</td><td class="cm-selected"><strong>8</strong> TP</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">recall = 8 / (8 + 2) = 80%</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> the same classifier has 80% recall but only 8% precision. The matrix reveals that abundant negatives create many more false alarms than true alerts.</figcaption>
</figure>

All binary classification metrics are derived from these four numbers:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | $(TP + TN) / N$ | Overall correctness |
| Precision (PPV) | $TP / (TP + FP)$ | Of predicted positives, fraction correct |
| Recall (TPR, Sensitivity) | $TP / (TP + FN)$ | Of actual positives, fraction caught |
| Specificity (TNR) | $TN / (TN + FP)$ | Of actual negatives, fraction correctly rejected |
| F1 | $2 PR / (P + R)$ | Harmonic mean of precision and recall |
| FPR | $FP / (FP + TN)$ | False alarm rate; $1 - \text{specificity}$ |
| FNR | $FN / (FN + TP)$ | Miss rate; $1 - \text{recall}$ |
| NPV | $TN / (TN + FN)$ | Negative predictive value |
| Balanced accuracy | $(TPR + TNR) / 2$ | Average of recall on each class |
| Matthews Corr Coef | see below | Single-number summary that handles imbalance |

**Matthews Correlation Coefficient (MCC)**:

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}.
$$

Range: $[-1, +1]$. $0$ = random; $+1$ = perfect; $-1$ = inverse-perfect. **Robust to class imbalance** in a way accuracy is not.

## Multi-class confusion matrix

For $K$ classes: $K \times K$ matrix. Diagonal entries are correct predictions; off-diagonal are confusions.

Per-class metrics: treat class $k$ as positive, all others as negative; compute binary metrics from the row and column sums.

Aggregate to system metrics by:

- **Macro**: average per-class metric (treats all classes equally).
- **Micro**: aggregate counts across classes (dominated by majority).
- **Weighted**: macro weighted by support.

## Visualizing the confusion matrix

Always visualize on multi-class problems:

- Heat-map with diagonal highlighted.
- Normalize by row (recall per class) or by column (precision per class). Choose based on what you need to debug.
- Sort rows/columns by frequency or by similarity for easier reading.

## Threshold dependence

For probabilistic classifiers, the confusion matrix depends on the decision threshold. As you sweep the threshold from 0 to 1:

- TP and FP both decrease (fewer predicted positives).
- The confusion matrix sweeps through different $(TP, FP, FN, TN)$ tuples.
- This is what produces the ROC and PR curves.

For deployment, pick the threshold that optimizes your real-world objective (precision at a recall floor, expected utility, etc.).

## What to report

A complete classification report should include:

1. **Confusion matrix** (raw counts and row-normalized).
2. **Per-class precision, recall, F1**.
3. **Macro-averaged metrics** for overall performance.
4. **One operating-point metric** (e.g., accuracy at threshold 0.5, or P@R=0.95).
5. **Threshold-free curves**: ROC-AUC and PR-AUC.

scikit-learn's `classification_report` and `confusion_matrix` produce most of this; pair with `roc_auc_score` and `average_precision_score`.

## Common pitfalls

- **Reporting a single number for a multi-class problem.** Always include the per-class breakdown.
- **Confusion matrix rows-vs-columns confusion.** Different libraries / textbooks use different conventions; explicitly label "rows = true" or "rows = predicted" in plots.
- **Computing precision/recall on training data.** Always evaluate on held-out data.
- **Choosing threshold by maximizing F1 on the test set.** That's tuning on test; use validation.
- **Treating per-class F1 as comparable across classes with very different prevalences.** Macro-average can mask huge per-class disparities; look at the table.

## Related

- [Precision, recall, F1](/concepts/precision-recall-f1/). Single-metric details.
- [ROC, PR, AUC](/concepts/roc-pr-auc/). Threshold-free metrics.
