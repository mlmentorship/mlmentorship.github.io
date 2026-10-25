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

Select the highest-score group first, then order that smaller group using original indices.

## Problem

For finite scores `[0.1, 0.9, 0.4, 0.8]`, return the two highest-score indices in descending score order.
The answer is `[1, 3]`, not sorted scores or positions within a selected subset.
The reference requires a one-dimensional NumPy array and integer `1 <= k <= n`.

## Picture

Partition selects membership `{1, 3}` without promising order.
Sort only their gathered scores, then map those local positions through `candidates`.
The trace uses one allowed candidate arrangement; NumPy may return another.

<!-- visual:top-k-scores-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="top-k-scores-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="top-k-scores-state-title">Top-K Scores: Use partial partition only to select top-k membership, then sort those candidate indices by their scores for the required descending order.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="top-k-scores" role="group" tabindex="0" aria-label="Top-K Scores: Use partial partition only to select top-k membership, then sort those candidate indices by their scores for the required descending order."><div class="coding-visual-example"><span>Input and goal</span><strong>Return indices of the `k` largest scores in descending score order.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="validate" role="group" aria-label="Validate k against score count"><div class="coding-trace-frame-heading"><span>Validate k against score count</span><strong>For four scores, k=2 passes validation.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="selection-cursor">input start</span><span class="coding-trace-array-cell">0.1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.9</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.8</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>guard</b>1 &lt;= k=2 &lt;= len(scores)=4</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="partition" hidden role="group" aria-label="Partition around the top-two boundary"><div class="coding-trace-frame-heading"><span>Partition around the top-two boundary</span><strong>np.argpartition(scores,-2)[-2:] selects indices 1 and 3 as an unordered candidate set; it does not promise their order.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="candidate-one">candidate index 1</span><span class="coding-trace-array-cell">0.9</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="candidate-three">candidate index 3</span><span class="coding-trace-array-cell">0.8</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>candidates</b>{1,3}</span><span><b>scoresAtCandidates</b>{0.9,0.8}</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="gather" hidden role="group" aria-label="Read only the selected scores"><div class="coding-trace-frame-heading"><span>Read only the selected scores</span><strong>Advanced indexing scores[candidates] gives the two values 0.9 and 0.8; all other scores leave the ordering work.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="candidate-one">selected index 1</span><span class="coding-trace-array-cell">index 1: 0.9</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="candidate-three">selected index 3</span><span class="coding-trace-array-cell">index 3: 0.8</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>subsetSize</b>k=2</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="sort-selected" hidden role="group" aria-label="Sort candidate positions by score"><div class="coding-trace-frame-heading"><span>Sort candidate positions by score</span><strong>argsort over [0.9,0.8] returns ascending positions [1,0]; reversing gives [0,1].</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="candidate-one">descending first</span><span class="coding-trace-array-cell">position 0 -&gt; index 1 -&gt; 0.9</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="candidate-three">descending second</span><span class="coding-trace-array-cell">position 1 -&gt; index 3 -&gt; 0.8</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>arithmetic</b>argsort=[1,0]; reverse=[0,1]</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="return" hidden role="group" aria-label="Index candidates in descending order"><div class="coding-trace-frame-heading"><span>Index candidates in descending order</span><strong>candidates[[0,1]] returns original score indices [1,3], whose values satisfy 0.9&gt;=0.8.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="candidate-one">score 0.9</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="candidate-three">score 0.8</span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>verification</b>scores[1]=0.9 &gt;= scores[3]=0.8</span><span><b>result</b>[1,3]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Validate k against score count</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Partition around the top-two boundary</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read only the selected scores</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Sort candidate positions by score</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Index candidates in descending order</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use argpartition when k ranked elements are needed from a much larger score vector and fully sorting all n scores is unnecessary.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After partition, every selected candidate belongs to the top-k score group although candidate order is unspecified; after subset argsort reversal, those same indices are descending by score.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Separate selection from ordering: first isolate the small set that can contain the answer, then pay sorting cost only on that set.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For four scores, k=2 passes validation. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

Every selected score is at least every excluded score; equal-score membership is unspecified.
Subset sorting adds descending order without changing membership.

## Code

```python
import numpy as np

def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
   if not 1 <= k <= len(scores):
      raise ValueError("k must name an item in scores")
   candidates = np.argpartition(scores, -k)[-k:]
   return candidates[np.argsort(scores[candidates])[::-1]]
```

Selection plus subset sorting uses $O(n+k\log k)$ work.
Peak index storage is $O(n)$: `argpartition` returns all indices, although the output contains only $k$.

## Change ties or input access

### 1. Choose earlier indices at the boundary

**New task:** break all equal-score ties by smaller original index; allow `k=0`.

**Picture and change:** select scores above the threshold, then fill from equal-score indices in input order.

```python
def top_k_stable_indices(scores, count):
   scores = np.asarray(scores, dtype=float)
   if scores.ndim != 1 or not np.isfinite(scores).all():
      raise ValueError("expected finite score vector")
   if not 0 <= count <= len(scores):
      raise ValueError("invalid count")
   if count == 0:
      return np.empty(0, dtype=int)
   threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
   above = np.flatnonzero(scores > threshold)
   equal = np.flatnonzero(scores == threshold)[:count - len(above)]
   selected = np.concatenate((above, equal))
   return selected[np.lexsort((selected, -scores[selected]))]
```

**Check:** `[5, 5, 5]`, count `2`, returns `[0, 1]`.

**Use when:** reproducible boundary membership matters.

### 2. Read a stream once

**New task:** retain only $O(k)$ candidates from finite scores.

**Picture and change:** keep a min-heap of `(score, -index)`; its root is the worst retained candidate.

```python
import heapq
import math

def streaming_top_k_indices(values, count):
   if count < 0:
      raise ValueError("negative count")
   if count == 0:
      return []
   heap = []
   for index, score in enumerate(values):
      if not math.isfinite(score):
         raise ValueError("expected finite score")
      candidate = score, -index
      if len(heap) < count:
         heapq.heappush(heap, candidate)
      elif candidate > heap[0]:
         heapq.heapreplace(heap, candidate)
   if len(heap) != count:
      raise ValueError("too few scores")
   return [-negative_index for score, negative_index in sorted(heap, reverse=True)]
```

**Check:** the fixture returns `[1, 3]`; three equal scores retain indices `[0, 1]`.

**Use when:** bounded retained state outweighs partition's vectorized selection.

## Transfer: why stable sorting alone fails

Would stably sorting an arbitrary tied candidate subset guarantee earlier boundary indices?

<details>
<summary>Check your answer</summary>

No. An earlier tied index might already be excluded.
Choose boundary membership before ordering; sorting cannot restore a missing item.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Index and score contracts

Returning local `argsort` positions answers a different question from returning original score indices.
Always apply the local permutation through `candidates`.

The reference does not specify equal-score membership or order.
Both adaptations choose descending score, then ascending original index.
Their counts must be integers.
The stable array variant converts scores to float64; inputs must retain their intended order and equality under that conversion.
Do not silently use NaN ordering as a ranking policy.

Stable threshold selection still uses $O(n)$ temporary arrays.
The stream uses $O(k)$ retained storage and $O(n\log(k+1)+k\log k)$ work.
Its zero-count case returns immediately without consuming the iterable.
For positive counts, insufficient scores raise an error.

Neither asymptotic analysis nor this trace proves partition faster for every size, dtype or platform.

</div>
