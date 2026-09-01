---
title: "Cross-Entropy From Logits"
description: "Compute mean multiclass cross-entropy from logits and integer labels."
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

> Compute mean multiclass cross-entropy from logits and integer labels.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:cross-entropy-from-logits-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="cross-entropy-from-logits-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="cross-entropy-from-logits-state-title">Cross-Entropy From Logits: Cross-entropy from logits is a stable log-sum-exp minus the selected logit.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="cross-entropy-from-logits" role="group" aria-label="Cross-Entropy From Logits: Cross-entropy from logits is a stable log-sum-exp minus the selected logit."><div class="coding-visual-example"><span>Input and goal</span><strong>Compute mean multiclass cross-entropy from logits and integer labels.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Choose the correct class"><div class="coding-trace-frame-heading"><span>Choose the correct class</span><strong>For logits [2,1,0], label 0 selects logit 2.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">correct</span><span class="coding-trace-array-cell">class 0: 2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">class 1: 1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">class 2: 0</span></span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Compute the normalizer"><div class="coding-trace-frame-heading"><span>Compute the normalizer</span><strong>logsumexp summarizes all class logits without building probabilities first.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">selected</span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span></span></div><div class="coding-trace-meta"><span><b>formula</b>log(exp(2)+exp(1)+exp(0))</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Subtract the correct logit"><div class="coding-trace-frame-heading"><span>Subtract the correct logit</span><strong>Loss = logsumexp(row) - 2 = 0.4076.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">normalizer</span><span class="coding-trace-array-cell">logsumexp(row)</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">subtract</span><span class="coding-trace-array-cell">correct logit 2</span></span></div><div class="coding-trace-meta"><span><b>result</b>0.4076</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Choose the correct class</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Compute the normalizer</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Subtract the correct logit</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Cross-entropy from logits is a stable log-sum-exp minus the selected logit.</p></div><figcaption><strong>Read it this way:</strong> For logits [2,1,0], label 0 selects logit 2. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Stable log-softmax plus indexed selection.

**Simple idea:** Subtract each row maximum. Compute the log normalizer for each example.
Subtract the correct-class logit, then average.

```python
import numpy as np

def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
   shifted = logits - np.max(logits, axis=1, keepdims=True)
   log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
   correct_logits = shifted[np.arange(len(labels)), labels]
   return float(np.mean(log_normalizer - correct_logits))
```

**Cost:** $O(batch \times classes)$ time and output-sized temporary space.
