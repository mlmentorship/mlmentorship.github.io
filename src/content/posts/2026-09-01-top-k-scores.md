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
<figure class="learning-figure coding-visual-figure" aria-labelledby="top-k-scores-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="top-k-scores-state-title">Top-K Scores: Partition finds membership in the top group; sort only that group for output order.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="top-k-scores" role="group" tabindex="0" aria-label="Top-K Scores: Partition finds membership in the top group; sort only that group for output order."><div class="coding-visual-example"><span>Input and goal</span><strong>Return indices of the `k` largest scores in descending score order.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Partition the scores"><div class="coding-trace-frame-heading"><span>Partition the scores</span><strong>Scores 0.9 and 0.8 belong to the top-2 group.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-candidate">candidate</span><span class="coding-trace-array-cell">0.9</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="marker-candidate">candidate</span><span class="coding-trace-array-cell">0.8</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>action</b>argpartition</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Sort selected candidates"><div class="coding-trace-frame-heading"><span>Sort selected candidates</span><strong>Only selected indices 1 and 3 need final ordering.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-first">first</span><span class="coding-trace-array-cell">index 1: 0.9</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-second">second</span><span class="coding-trace-array-cell">index 3: 0.8</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>action</b>sort k</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return indices"><div class="coding-trace-frame-heading"><span>Return indices</span><strong>The descending top-k indices are [1,3].</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-top">top</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-top">top</span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>result</b>[1,3]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Partition the scores</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Sort selected candidates</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return indices</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Partition finds membership in the top group; sort only that group for output order.</p></div><figcaption><strong>Read it this way:</strong> Scores 0.9 and 0.8 belong to the top-2 group. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
