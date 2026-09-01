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
<figure class="learning-figure coding-visual-figure" aria-labelledby="cross-entropy-from-logits-state-title"><p class="visual-kicker">Stable numerical path</p><p class="visual-title" id="cross-entropy-from-logits-state-title">Cross-Entropy From Logits: Change the reference point before exponentiating</p><div class="coding-visual coding-visual--numerics" data-coding-visual data-coding-mode="numerics" data-coding-slug="cross-entropy-from-logits" role="group" aria-label="Cross-Entropy From Logits: loss = logsumexp(row) - the selected class logit. Subtracting one row constant changes no softmax probabilities or cross-entropy differences."><div class="coding-visual-example"><span>Concrete trace</span><strong>loss = logsumexp(row) - the selected class logit</strong></div><div class="coding-visual-sketch coding-visual-sketch--numerics"><div class="coding-sketch-array"><span class="coding-sketch-cell coding-sketch-cell--state">large</span><span class="coding-sketch-arrow">&minus; max</span><span class="coding-sketch-cell coding-sketch-cell--active">small</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">safe exp</span></div><p class="coding-sketch-note">relative differences stay; raw magnitude stops causing overflow</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Anchor</span><strong>row maximum</strong><small>Choose a value that keeps shifted logits small.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Shift</span><strong>logits − max</strong><small>Preserve relative differences while avoiding overflow.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Normalize</span><strong>logsumexp or softmax</strong><small>Aggregate the shifted exponentials stably.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Select</span><strong>correct class</strong><small>Read the requested probability or loss term.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Subtracting one row constant changes no softmax probabilities or cross-entropy differences.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The large raw numbers are not the information. Their differences are. Shift the row first, then exponentiate values that are numerically safe. For this problem, hold onto the concrete trace: loss = logsumexp(row) - the selected class logit.</figcaption></figure>

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
