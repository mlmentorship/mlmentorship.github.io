---
title: "Top-K Scores"
description: "Return indices of the `k` largest scores in descending score order."
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

> Return indices of the `k` largest scores in descending score order.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:top-k-scores-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="top-k-scores-state-title"><p class="visual-kicker">Select first, sort last</p><p class="visual-title" id="top-k-scores-state-title">Top-K Scores: Avoid ordering values that cannot enter the answer</p><div class="coding-visual coding-visual--selection" data-coding-visual data-coding-mode="selection" data-coding-slug="top-k-scores" role="group" aria-label="Top-K Scores: argpartition finds the top group; only those k scores receive final sorting. The retained group contains the k largest values before final ordering."><div class="coding-visual-example"><span>Concrete trace</span><strong>argpartition finds the top group; only those k scores receive final sorting</strong></div><div class="coding-visual-sketch coding-visual-sketch--selection"><div class="coding-sketch-array"><span class="coding-sketch-cell">discard</span><span class="coding-sketch-cell coding-sketch-cell--state">candidate</span><span class="coding-sketch-cell coding-sketch-cell--state">candidate</span><span class="coding-sketch-cell coding-sketch-cell--active">top k</span></div><p class="coding-sketch-note">membership first, presentation order second</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Partition</span><strong>candidate boundary</strong><small>Separate a possible top-k group without full sorting.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Retain</span><strong>k candidates</strong><small>Discard everything that cannot be selected.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Order</span><strong>selected scores</strong><small>Sort only the retained group.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Return</span><strong>indices + scores</strong><small>Emit the requested order and shape.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The retained group contains the k largest values before final ordering.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Selection answers membership; sorting answers presentation order. Keeping those jobs separate is the memory and time saving. For this problem, hold onto the concrete trace: argpartition finds the top group; only those k scores receive final sorting.</figcaption></figure>

**Pattern:** Partial selection, then sort only the selected values.

**Simple idea:** `argpartition` finds the top group without sorting every score. Sort the
small selected group for final order.

```python
import numpy as np

def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
   if not 1 <= k <= len(scores):
      raise ValueError("k must name an item in scores")
   candidates = np.argpartition(scores, -k)[-k:]
   return candidates[np.argsort(scores[candidates])[::-1]]
```

**Cost:** $O(n + k\log k)$ average time and $O(k)$ selected space.
