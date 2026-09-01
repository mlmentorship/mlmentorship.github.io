---
title: "Precision, recall, and F1"
description: "The three metrics every classifier interview asks about. Their definitions, when to optimize which, and the F-beta generalization."
date: "2025-09-20"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

For a binary classifier:

- **Precision** = $\frac{TP}{TP + FP}$. Of the positives I predicted, what fraction are correct?
- **Recall** = $\frac{TP}{TP + FN}$. Of the actual positives, what fraction did I catch?
- **F1** = $\frac{2 \cdot P \cdot R}{P + R}$. Harmonic mean of the two.

These are the three most-cited classification metrics. Picking the wrong one for your problem is a senior-level mistake; recommending "use precision and recall" without specifying the operating point is a common interview tell.

## The four base counts

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP | FN |
| **Actually negative** | FP | TN |

Different metrics weight these differently:

- **Accuracy** = $(TP + TN) / N$. Fraction correct overall. Useless on imbalanced data.
- **Precision** = $TP / (TP + FP)$. Column purity (predicted positive).
- **Recall** = $TP / (TP + FN)$. Row purity (actual positive). Also called **sensitivity** or **true positive rate**.
- **Specificity** = $TN / (TN + FP)$. True negative rate.

## When to favor precision vs. recall

Choose based on the relative cost of false positives vs. false negatives:

| Scenario | Cost of FP | Cost of FN | Optimize |
|----------|-----------|-----------|----------|
| Spam filter | User loses important email | User sees spam | Precision |
| Cancer screening | Unnecessary biopsy | Missed cancer | Recall |
| Web search top result | Wrong page surfaces | Right page on page 2 | Precision |
| Fraud detection | Legitimate transaction blocked | Fraud succeeds | Both. Depends on dollar values |
| Recommendation candidate generation | Boring rec | Missing a perfect rec | Recall (filter later) |

There is **always a tradeoff**: increasing one decreases the other along the precision-recall curve. The right choice depends on the operating cost.

## F1 and F-beta

The F1 score balances precision and recall via the harmonic mean. **Harmonic mean penalizes imbalance**: F1 is low if either P or R is low, even if the other is high.

<!-- visual:precision-recall-f1-harmonic-penalty -->
<figure class="learning-figure" aria-labelledby="f1-harmonic-penalty-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="f1-harmonic-penalty-title">See why F1 falls toward the smaller metric instead of rewarding the arithmetic average.</p>
	<div class="visual-grid--two">
		<section class="visual-panel" aria-labelledby="f1-balanced-title">
			<h4 id="f1-balanced-title">Balanced pair</h4>
			<p>Neither metric hides a weak side.</p>
			<table class="cm-grid" aria-label="Balanced precision and recall both equal 0.50, producing arithmetic mean 0.50 and F1 0.50">
				<tbody>
					<tr><th scope="row">Precision</th><td><strong>0.50</strong></td></tr>
					<tr><th scope="row">Recall</th><td><strong>0.50</strong></td></tr>
					<tr><th scope="row">Arithmetic mean</th><td><strong>0.50</strong></td></tr>
					<tr><th class="cm-selected" scope="row">F1</th><td class="cm-selected"><strong>0.50</strong></td></tr>
				</tbody>
			</table>
			<p class="cm-equation">2(0.50)(0.50) / (0.50 + 0.50) = 0.50</p>
		</section>
		<section class="visual-panel" aria-labelledby="f1-imbalanced-title">
			<h4 id="f1-imbalanced-title">Imbalanced pair</h4>
			<p>The same arithmetic mean masks one weak metric.</p>
			<table class="cm-grid" aria-label="Imbalanced precision 0.90 and recall 0.10 produce arithmetic mean 0.50 but F1 only 0.18">
				<tbody>
					<tr><th scope="row">Precision</th><td><strong>0.90</strong></td></tr>
					<tr><th scope="row">Recall</th><td><strong>0.10</strong></td></tr>
					<tr><th scope="row">Arithmetic mean</th><td><strong>0.50</strong></td></tr>
					<tr><th class="cm-selected" scope="row">F1</th><td class="cm-selected"><strong>0.18</strong></td></tr>
				</tbody>
			</table>
			<p class="cm-equation">2(0.90)(0.10) / (0.90 + 0.10) = 0.18</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> both pairs average to 0.50 arithmetically, but F1 drops to 0.18 when recall is only 0.10. A high precision score cannot compensate for weak recall (and vice versa).</figcaption>
</figure>

The **F-beta** generalization weights recall $\beta^2$ times more than precision:

$$
F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}.
$$

- $\beta = 1$ → F1 (equal weight).
- $\beta = 2$ → F2 (recall weighted 4× more. Favors finding all positives).
- $\beta = 0.5$ → F0.5 (precision weighted 4× more. Favors avoiding false positives).

## Multi-class

For $K$ classes, two averaging strategies:

- **Macro-averaged**: compute precision/recall/F1 per class, then average. Treats all classes equally. Penalizes models that ignore minority classes.
- **Micro-averaged**: aggregate TP/FP/FN across all classes, then compute. Equivalent to overall accuracy for multi-class. Dominated by majority classes.
- **Weighted**: macro-average weighted by class support. Compromise.

For imbalanced multi-class: report **macro-F1** to ensure minority classes are evaluated, and **per-class precision-recall** for diagnosis.

## Threshold dependence

Precision and recall are computed at a specific decision threshold (e.g., 0.5 for predicted probability). A single P/R pair represents one operating point on the **precision-recall curve**. Always know which threshold you used.

For threshold-independent comparison, report:

- **PR-AUC** (area under the precision-recall curve).
- **ROC-AUC** (separability metric, threshold-free).
- See [ROC, PR, and AUC](/concepts/roc-pr-auc/).

## Common pitfalls

- **Reporting accuracy on imbalanced data.** A 1% positive class trivially gets 99% accuracy by predicting "negative" always.
- **Reporting F1 alone.** F1 is a single number; the precision-recall tradeoff matters.
- **Comparing F1 across datasets with different positive priors.** F1 depends on class balance.
- **Treating threshold 0.5 as default.** It's an arbitrary choice; pick from the PR curve at the deployment operating point.
- **Confusing recall with sensitivity (and specificity).** Recall = sensitivity = TPR. Specificity is a *different* quantity (TNR).
- **Macro-F1 on label-skewed data**: a model that aces 99 majority-class examples and bombs 1 minority class still gets a poor macro-F1. Sometimes that's the right signal, sometimes it's misleading.

## Related

- [ROC, PR, and AUC](/concepts/roc-pr-auc/). Threshold-independent metrics.
- [Confusion matrix](/concepts/confusion-matrix-and-classification-metrics/). Full classification metric reference.
- [Class imbalance](/questions/class-imbalance/). How class skew affects metric choice.
