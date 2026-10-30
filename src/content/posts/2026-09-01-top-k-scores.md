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

Return the original positions of the highest scores, ordered from highest selected score to lowest.

## Problem

**Input.** A one-dimensional NumPy array `scores` of finite values and an integer `k` with `1 <= k <= len(scores)`.

**Output.** Return an array of `k` distinct original indices, in nonincreasing order of their corresponding scores.

**Rules.** Return input positions, not scores or positions within a selected subset.
At a tied cutoff, any highest-scoring membership is valid; equal-score indices need no particular order.
Preserve the input. Empty input and out-of-range `k` raise `ValueError` in the reference.

**Example.** Scores `[0.1, 0.9, 0.4, 0.8]` with `k = 2` return `[1, 3]`.
Those original positions contain 0.9 and 0.8, respectively.

## Picture

Partition selects membership `{1, 3}` without promising order.
Sort only their gathered scores, then map those local positions through `candidates`.
The trace uses one allowed candidate arrangement; NumPy may return another.

<!-- visual:top-k-scores-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="top-k-scores-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="top-k-scores-state-title">Map candidate-local sort positions back to original score indices</p><div class="visual-argument" data-visual-argument="top-k-scores-state" data-argument-kind="ledger"><p class="visual-argument-premise">Worked example: partition selects membership; it does not sort the selected candidates</p><svg viewBox="0 0 360 322" role="img" aria-label="Worked example: partition selects membership; it does not sort the selected candidates. Candidate indices (3,1) have correct membership but no score order. Sorting that subset returns indices (1,3)."><title>Worked example: partition selects membership; it does not sort the selected candidates</title><desc>Candidate indices (3,1) have correct membership but no score order. Sorting that subset returns indices (1,3).</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Original score order</tspan></text><rect data-proof-token=".2" x="20" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="58" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">.2</tspan></text><rect data-proof-token=".9" x="100" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="138" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">.9</tspan></text><rect data-proof-token=".4" x="180" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="218" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">.4</tspan></text><rect data-proof-token=".8" x="260" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="298" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">.8</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">One permitted unordered top-2 subset</tspan></text><rect data-proof-token="3: .8" x="20" y="156" width="156" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="98" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">3: .8</tspan></text><rect data-proof-token="1: .9" x="180" y="156" width="156" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="258" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">1: .9</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Descending-by-score original indices</tspan></text><rect data-proof-token="1" x="20" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="98" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">1</tspan></text><rect data-proof-token="3" x="180" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="258" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">3</tspan></text></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="top-k-scores" role="group" tabindex="0" aria-label="Top-K Scores: Use partial partition only to select top-k membership, then sort those candidate indices by their scores for the required descending order."><div class="coding-visual-example"><span>Input and goal</span><strong>Return indices of the `k` largest scores in descending score order.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="validate" role="group" aria-label="Validate k against score count"><div class="coding-trace-frame-heading"><span>Validate k against score count</span><strong>For four scores, k=2 passes validation.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="selection-cursor">input start</span><span class="coding-trace-array-cell">0.1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.9</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.8</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>guard</b>1 &lt;= k=2 &lt;= len(scores)=4</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>input start</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>guard</dt><dd><strong class="coding-state-after">1 &lt;= k=2 &lt;= len(scores)=4</strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><strong class="coding-state-after">0.1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="partition" hidden role="group" aria-label="Partition around the top-two boundary"><div class="coding-trace-frame-heading"><span>Partition around the top-two boundary</span><strong>np.argpartition(scores,-2)[-2:] selects indices 1 and 3 as an unordered candidate set; it does not promise their order.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="candidate-one">candidate index 1</span><span class="coding-trace-array-cell">0.9</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="candidate-three">candidate index 3</span><span class="coding-trace-array-cell">0.8</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>candidates</b>{1,3}</span><span><b>scoresAtCandidates</b>{0.9,0.8}</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>candidate index 1</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>candidate index 3</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>input start</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="gather" hidden role="group" aria-label="Read only the selected scores"><div class="coding-trace-frame-heading"><span>Read only the selected scores</span><strong>Advanced indexing scores[candidates] gives the two values 0.9 and 0.8; all other scores leave the ordering work.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="candidate-one">selected index 1</span><span class="coding-trace-array-cell">index 1: 0.9</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="candidate-three">selected index 3</span><span class="coding-trace-array-cell">index 3: 0.8</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>subsetSize</b>k=2</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>selected index 1</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>selected index 3</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>subset size</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">k=2</strong></dd></div></dl><div class="coding-numeric-transition"><svg viewBox="0 0 360 266" role="img" aria-label="selected index 1: 1 before, 0 at the next shown checkpoint. selected index 3: 3 before, 1 at the next shown checkpoint." style="display:block;width:100%;max-width:34rem;margin:auto"><title>selected index 1: 1 before, 0 at the next shown checkpoint. selected index 3: 3 before, 1 at the next shown checkpoint.</title><desc>Each row has its own zero-inclusive numerical scale. Circles mark the preceding authored checkpoint and squares the current checkpoint. Connecting segments compare these shown values, not an invented path of execution.</desc><text x="16" y="20" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text-soft)">SHOWN CHECKPOINTS</text><text x="344" y="20" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text-soft)">circle: before / square: now</text><g data-numeric-transition="pointer:candidate-one" data-before="1" data-after="0" data-minimum="0" data-maximum="1"><text x="16" y="45" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">selected index 1</text><text x="344" y="45" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">1 -&gt; 0</text><path d="M28 96H332" style="stroke:var(--c-rule);fill:none"></path><path d="M28 65V111" style="stroke:var(--c-text-soft);stroke-dasharray:3 3;fill:none"></path><path d="M332 69L28 96" style="stroke:var(--viz-focus-stroke);stroke-width:2;fill:none"></path><circle data-checkpoint-before cx="332" cy="69" r="5" style="fill:var(--c-bg);stroke:var(--c-text-soft);stroke-width:2"></circle><rect data-checkpoint-after x="23" y="91" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="28" y="128" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">0</text><text x="332" y="128" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">1</text></g><g data-numeric-transition="pointer:candidate-three" data-before="3" data-after="1" data-minimum="0" data-maximum="3"><text x="16" y="153" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">selected index 3</text><text x="344" y="153" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">3 -&gt; 1</text><path d="M28 204H332" style="stroke:var(--c-rule);fill:none"></path><path d="M28 173V219" style="stroke:var(--c-text-soft);stroke-dasharray:3 3;fill:none"></path><path d="M332 177L129.33333333333331 204" style="stroke:var(--viz-focus-stroke);stroke-width:2;fill:none"></path><circle data-checkpoint-before cx="332" cy="177" r="5" style="fill:var(--c-bg);stroke:var(--c-text-soft);stroke-width:2"></circle><rect data-checkpoint-after x="124.33333333333331" y="199" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="28" y="236" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">0</text><text x="332" y="236" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">3</text></g></svg></div></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="sort-selected" hidden role="group" aria-label="Sort candidate positions by score"><div class="coding-trace-frame-heading"><span>Sort candidate positions by score</span><strong>argsort over [0.9,0.8] returns ascending positions [1,0]; reversing gives [0,1].</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="candidate-one">descending first</span><span class="coding-trace-array-cell">position 0 -&gt; index 1 -&gt; 0.9</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="candidate-three">descending second</span><span class="coding-trace-array-cell">position 1 -&gt; index 3 -&gt; 0.8</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>arithmetic</b>argsort=[1,0]; reverse=[0,1]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>arithmetic</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">argsort=[1,0]; reverse=[0,1]</strong></dd></div><div class="coding-state-change"><dt>subset size</dt><dd><span class="coding-state-before">k=2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><span class="coding-state-before">index 1: 0.9</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">position 0 -&gt; index 1 -&gt; 0.9</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="return" hidden role="group" aria-label="Index candidates in descending order"><div class="coding-trace-frame-heading"><span>Index candidates in descending order</span><strong>candidates[[0,1]] returns original score indices [1,3], whose values satisfy 0.9&gt;=0.8.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="candidate-one">score 0.9</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="candidate-three">score 0.8</span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>verification</b>scores[1]=0.9 &gt;= scores[3]=0.8</span><span><b>result</b>[1,3]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>verification</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">scores[1]=0.9 &gt;= scores[3]=0.8</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[1,3]</strong></dd></div><div class="coding-state-change"><dt>arithmetic</dt><dd><span class="coding-state-before">argsort=[1,0]; reverse=[0,1]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Validate k against score count</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Partition around the top-two boundary</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read only the selected scores</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Sort candidate positions by score</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Index candidates in descending order</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use argpartition when k ranked elements are needed from a much larger score vector and fully sorting all n scores is unnecessary.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After partition, every selected candidate belongs to the top-k score group although candidate order is unspecified; after subset argsort reversal, those same indices are descending by score.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Separate selection from ordering: first isolate the small set that can contain the answer, then pay sorting cost only on that set.</dd></div></dl></div><div class="visual-derivation-caption">For four scores, k=2 passes validation. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="top-k-scores-state"><strong>Check:</strong> Why must the selected scores' argsort positions index the candidate array before returning original score indices?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="top-k-scores-state">Candidate indices (3,1) have correct membership but no score order. Sorting that subset returns indices (1,3).</span></figcaption></figure>

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
