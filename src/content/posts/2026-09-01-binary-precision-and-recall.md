---
title: "Binary Precision and Recall"
description: "Compute precision and recall from binary labels and predictions."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Role-specific"
aliases: []
prerequisites: []
---

> Compute precision and recall from binary labels and predictions.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:binary-precision-and-recall-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-precision-and-recall-state-title"><p class="visual-kicker">Counts before ratios</p><p class="visual-title" id="binary-precision-and-recall-state-title">Binary Precision and Recall: Accumulate the four cells, then compute precision and recall</p><div class="coding-visual coding-visual--metrics" data-coding-visual data-coding-mode="metrics" data-coding-slug="binary-precision-and-recall" role="group" aria-label="Binary Precision and Recall: TP=2, FP=1, FN=1 -&gt; precision uses predicted positives; recall uses real positives. Each observation contributes to one and only one confusion-matrix count."><div class="coding-visual-example"><span>Concrete trace</span><strong>TP=2, FP=1, FN=1 -&gt; precision uses predicted positives; recall uses real positives</strong></div><div class="coding-visual-sketch coding-visual-sketch--metrics"><div class="coding-sketch-matrix coding-sketch-matrix--metrics"><span class="coding-sketch-grid-cell">TN</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">FP</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">FN</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">TP</span></div><p class="coding-sketch-note">precision reads the predicted-positive column; recall reads the actual-positive row</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Classify</span><strong>truth × prediction</strong><small>Send each example to exactly one confusion cell.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Count</span><strong>TP FP TN FN</strong><small>Retain additive counts rather than raw examples.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Divide</span><strong>chosen denominator</strong><small>Use predicted positives for precision or real positives for recall.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Guard</span><strong>zero denominator</strong><small>Define the empty-class behavior explicitly.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each observation contributes to one and only one confusion-matrix count.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The denominator is part of the metric. Draw the count cells first, then draw the ratio that selects the row or column it needs. For this problem, hold onto the concrete trace: TP=2, FP=1, FN=1 -&gt; precision uses predicted positives; recall uses real positives.</figcaption></figure>

**Pattern:** Boolean masks and safe division.

**Simple idea:** Count true positives, false positives, and false negatives with Boolean
array operations. Return zero when a denominator is zero.

```python
import numpy as np

def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
   true_positive = int(np.sum((labels == 1) & (predictions == 1)))
   false_positive = int(np.sum((labels == 0) & (predictions == 1)))
   false_negative = int(np.sum((labels == 1) & (predictions == 0)))

   precision_total = true_positive + false_positive
   recall_total = true_positive + false_negative
   precision = true_positive / precision_total if precision_total else 0.0
   recall = true_positive / recall_total if recall_total else 0.0
   return {"precision": precision, "recall": recall}
```

**Cost:** $O(n)$ time and temporary Boolean arrays.
