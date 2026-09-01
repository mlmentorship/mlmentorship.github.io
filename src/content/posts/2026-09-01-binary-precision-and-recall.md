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
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-precision-and-recall-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="binary-precision-and-recall-state-title">Binary Precision and Recall: Each example enters one confusion cell; the metric chooses its denominator afterward.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="binary-precision-and-recall" role="group" aria-label="Binary Precision and Recall: Each example enters one confusion cell; the metric chooses its denominator afterward."><div class="coding-visual-example"><span>Input and goal</span><strong>Compute precision and recall from binary labels and predictions.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Classify observations"><div class="coding-trace-frame-heading"><span>Classify observations</span><strong>Truth and prediction route examples to TN, FP, FN, or TP.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col"></th><th scope="col">pred 0</th><th scope="col">pred 1</th></tr></thead><tbody><tr><td class="">true 0</td><td class="is-active">TN</td><td class="is-active">FP</td></tr><tr><td class="">true 1</td><td class="is-active">FN</td><td class="is-active">TP</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Count the cells"><div class="coding-trace-frame-heading"><span>Count the cells</span><strong>For the example, TP=1, FP=1, FN=1, TN=1.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col"></th><th scope="col">pred 0</th><th scope="col">pred 1</th></tr></thead><tbody><tr><td class="">true 0</td><td class="is-active">1 TN</td><td class="is-active">1 FP</td></tr><tr><td class="">true 1</td><td class="is-active">1 FN</td><td class="is-active">1 TP</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>counts</b>TP=1 FP=1 TN=1 FN=1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Choose the denominator"><div class="coding-trace-frame-heading"><span>Choose the denominator</span><strong>Precision uses predicted positives; recall uses actual positives.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">metric</th><th scope="col">numerator</th><th scope="col">denominator</th></tr></thead><tbody><tr><td class="">precision</td><td class="is-active">TP=1</td><td class="is-active">TP+FP=2</td></tr><tr><td class="">recall</td><td class="is-active">TP=1</td><td class="is-active">TP+FN=2</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>result</b>precision=.5, recall=.5</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Classify observations</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Count the cells</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Choose the denominator</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Each example enters one confusion cell; the metric chooses its denominator afterward.</p></div><figcaption><strong>Read it this way:</strong> Truth and prediction route examples to TN, FP, FN, or TP. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
