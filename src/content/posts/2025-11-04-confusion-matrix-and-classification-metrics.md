---
title: "Confusion matrix and classification metrics"
description: "A confusion matrix counts predicted labels against actual labels. Read precision, recall and other fixed-threshold metrics from their correct populations."
date: "2025-11-04"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A confusion matrix counts predicted labels against actual labels.
Rows and columns select different populations, so they answer different questions.

Suppose 1,000 held-out cases produce 98 alerts. Eight alerts are correct, and two positive cases receive no alert.
How many alerts are useful? How many actual positives did the model find?
Use mature labels and one fixed threshold for this example.

## The binary confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN (True Neg) | FP (False Pos, Type I) |
| **Actual 1** | FN (False Neg, Type II) | TP (True Pos) |

<!-- visual:confusion-matrix-denominators -->
<figure class="learning-figure" aria-labelledby="cm-denominators-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="cm-denominators-title">Eight hits can mean high recall and low precision</p>
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
			<p class="cm-equation">precision = 8 / (8 + 90) &asymp; 8.16%</p>
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
	<figcaption><strong>Read it this way:</strong> the same classifier has 80% recall and about 8.16% precision. The matrix shows many more false alarms than true alerts. <span class="visual-reader-check" data-reader-check="confusion-matrix-denominators"><strong>Check:</strong> How can the same eight true positives yield 80% recall but only about 8.16% precision?</span></figcaption>
</figure>

Read the diagram's positive prediction column: 90 false alerts plus eight true alerts.
Precision is $8/98\approx0.08163$, or about 8.16%.
Read the actual-positive row: two misses plus eight true alerts.
Recall is $8/10=0.8$.

The remaining 900 cases are true negatives.
Check the total: $900+90+2+8=1000$.
Accuracy is $908/1000=0.908$, although most alerts are false.

The denominator must count the population named in the question.
Let $N=TP+TN+FP+FN$. These metrics use the four counts at this threshold:

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

When its denominator is nonzero, MCC ranges from $-1$ to $1$.
Perfect agreement gives $1$; complete reversal gives $-1$.
Zero means no observed binary correlation, not proof of a useful or random model.
MCC uses all four marginal totals. It is not invariant to class prevalence.

For this table, $F_1=16/(16+90+2)\approx0.14815$.
Specificity is $900/990\approx0.90909$.
Balanced accuracy is $(0.8+0.90909)/2\approx0.85455$.
The table describes errors. Product costs determine whether those errors are acceptable.

## Change the population

### 1. Find which actual class is missed

**New task:** compare recall across actual classes.

**Picture and change:** divide each row by its own total, rather than dividing a prediction column.
Keep raw counts visible beside the normalized table.

**Check:** the actual-positive row becomes $[0.2,0.8]$.
The positive prediction column instead gives precision $8/98$.
They describe different populations and need not agree.

**Use when:** the question concerns coverage of an actual class.

### 2. Give each class equal weight

**New task:** prevent the majority class from dominating a recall summary.

**Picture and change:** calculate recall for each class, then average the two results.
Do not pool their counts first.

**Check:** equal weighting gives balanced accuracy, about $0.85455$.
Pooling correct predictions gives accuracy $0.908$, because 990 of 1,000 cases are negative.

**Use when:** each class has equal decision importance, regardless of its frequency.
Equal weighting still does not remove the need to inspect each class.

## What this table cannot tell you

Two models can produce identical labels at threshold $0.5$ but order their scores differently.
Their confusion matrices match, while ranking measures and log losses can differ.
Use scores to evaluate those properties. One thresholded table cannot recover them.

No predicted positives gives precision denominator zero. No actual positives gives recall denominator zero.
State the undefined-value convention and show the support. An assigned zero is a reporting policy, not an observed error rate.

## Try a missing population

There are no positive predictions. Can precision prove that the model's alerts are reliable?

<details>
<summary>Check your answer</summary>

No. There are no alerts to assess, so precision is undefined before applying a reporting convention.
Report zero predicted support beside that convention.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Multi-class confusion matrix

For $K$ classes, use a $K\times K$ table. Diagonal entries are correct predictions; other entries are confusions.
For class $k$, treat all other classes as negative and recover counts from row and column totals.

- **Macro:** average the per-class metrics equally.
- **Micro:** combine counts first, then calculate the metric.
- **Weighted:** weight each per-class metric by its support.

In ordinary single-label multiclass classification, micro precision and recall both equal accuracy.
Do not extend that identity to multilabel predictions.

## Visualizing the confusion matrix

Label both axes explicitly. Use row normalization for recall and column normalization for precision.
Keep counts available and use the same class order on both axes.

## Threshold dependence

Increasing the positive-score threshold cannot increase TP or FP on the same examples.
Precision can still move either way.
Threshold sweeps need the scores, not only one matrix.
Choose the operating point on validation data using the stated costs or constraints.

## What to report

Report counts, per-class metrics, the operating threshold and the aggregation convention.
Add score-based curves when they answer the decision question.
State whether PR area means average precision or another integration rule.
Scikit-learn provides `confusion_matrix`, `classification_report`, `roc_auc_score` and `average_precision_score`.

## Common pitfalls

- State the axis convention; libraries differ.
- Use held-out data, not training performance, for generalization claims.
- Do not select a threshold on the final test set.
- Inspect class supports. One average can hide a failed class.

## Related

- [Precision, recall, F1](/concepts/precision-recall-f1/). Single-metric details.
- [ROC, PR, AUC](/concepts/roc-pr-auc/). Threshold-free metrics.

</div>
