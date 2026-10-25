---
title: "Use bounded occurrence counts as ranking coordinates"
description: "Place distinct values at their frequency coordinates, then change memory use or specify exact cutoff ties."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Core"
aliases: []
prerequisites: []
---

Use each occurrence count as a bucket coordinate, then scan from the highest count downward.

## Problem

For `[1,1,1,2,2,3]` and `k = 2`, return **[1,2]**.
Candidates are distinct integer values; priority is frequency, not numeric size.
Use integer `k` with `0 <= k <= u`, where `u` is the distinct count.
Any cutoff-tied selection is valid unless a separate tie rule is specified.

## Picture

Imagine a row of coordinates labelled `0` through `n`.
A value's frequency tells you exactly where it belongs.
No positive count exceeds the input length `n`.

| Value | Frequency | Bucket |
| --- | --- | --- |
| `1` | `3` | `3` |
| `2` | `2` | `2` |
| `3` | `1` | `1` |

Skip empty buckets `6,5,4`, take value `1` from bucket `3`, then value `2` from bucket `2`.

<!-- visual:top-k-frequent-elements-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="top-k-frequent-elements-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="top-k-frequent-elements-state-title">Use bounded occurrence counts as ranking coordinates: Frequency is a bounded coordinate: place each value at its count, then scan counts from n down until k values are collected.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="top-k-frequent-elements" role="group" tabindex="0" aria-label="Use bounded occurrence counts as ranking coordinates: Frequency is a bounded coordinate: place each value at its count, then scan counts from n down until k values are collected."><div class="coding-visual-example"><span>Input and goal</span><strong>Return the `k` values that appear most often.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="count-values" role="group" aria-label="Count the concrete input"><div class="coding-trace-frame-heading"><span>Count the concrete input</span><strong>Value 1 appears three times, value 2 twice, and value 3 once. Select two distinct values.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>k</b>2</span><span><b>mapLabel</b>frequency</span></div><div class="coding-trace-map"><span class="coding-trace-label">frequency</span><span class="coding-trace-map-entry"><b>1</b><span>3</span></span><span class="coding-trace-map-entry"><b>2</b><span>2</span></span><span class="coding-trace-map-entry"><b>3</b><span>1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="build-buckets" hidden role="group" aria-label="Place values at frequency coordinates"><div class="coding-trace-frame-heading"><span>Place values at frequency coordinates</span><strong>Allocate n+1 = 7 buckets. Put value 3 at index 1, value 2 at index 2, and value 1 at index 3.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=0 []</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=1 [3]</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=2 [2]</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=3 [1]</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=4 []</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=5 []</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-pointer" data-motion-key="frequency-scan">scan starts</span><span class="coding-trace-array-cell">f=6 []</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>answer</b>[]</span><span><b>direction</b>6 -&gt; 0</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="skip-empty-buckets" hidden role="group" aria-label="Skip empty high frequencies"><div class="coding-trace-frame-heading"><span>Skip empty high frequencies</span><strong>Buckets 6, 5, and 4 are empty, so extending the answer changes nothing and the scan moves down to frequency 3.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=0 []</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=1 [3]</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=2 [2]</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="frequency-scan">scan f=3</span><span class="coding-trace-array-cell">f=3 [1]</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=4 []</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=5 []</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=6 []</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>answer</b>[]</span><span><b>skipped</b>f=6,5,4</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="take-frequency-3" hidden role="group" aria-label="Collect the most frequent value"><div class="coding-trace-frame-heading"><span>Collect the most frequent value</span><strong>Extend with bucket 3 = [1]. The answer is [1], whose length 1 is still below k = 2.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=0 []</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=1 [3]</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=2 [2]</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="frequency-scan">take [1]</span><span class="coding-trace-array-cell">f=3 [1]</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=4 []</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=5 []</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=6 []</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>answer</b>[1]</span><span><b>check</b>1 &lt; 2; continue</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="take-frequency-2" hidden role="group" aria-label="Collect until k is reached"><div class="coding-trace-frame-heading"><span>Collect until k is reached</span><strong>Move to bucket 2 = [2] and extend: [1] + [2] = [1,2]. Its length is now k, so return the first two values.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=0 []</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=1 [3]</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="frequency-scan">take [2]</span><span class="coding-trace-array-cell">f=2 [2]</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=3 [1]</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=4 []</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=5 []</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">f=6 []</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>answer</b>[1,2]</span><span><b>check</b>2 &gt;= k</span><span><b>result</b>[1, 2]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Count the concrete input</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Place values at frequency coordinates</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Skip empty high frequencies</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Collect the most frequent value</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Collect until k is reached</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use frequency buckets when the answer ranks values by occurrence count, counts cannot exceed the input length n, and linear time matters more than ordering values within equal-frequency buckets.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before scanning bucket f, the answer contains every value from frequencies greater than f and no value from a lower frequency. Therefore the first k collected values are among the k most frequent.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Whenever a ranking key is a small bounded integer, use that key as an array coordinate and scan coordinates in answer order; this transfers to counting sort, histogram selection, and bounded-score ranking.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Value 1 appears three times, value 2 twice, and value 3 once. Select two distinct values. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

Before scanning bucket `f`, all higher-frequency candidates have been collected; no lower-frequency candidate has been collected.
That ordering makes the first `k` collected values a valid selection.
A cutoff bucket can temporarily add more than `k`; slicing keeps exactly `k`.

## Code

```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
   frequencies = Counter(nums)
   buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]

   for num, frequency in frequencies.items():
      buckets[frequency].append(num)

   answer: list[int] = []
   for bucket in reversed(buckets):
      answer.extend(bucket)
      if len(answer) >= k:
         return answer[:k]
   return answer
```

Expected time and storage are $O(n+u)=O(n)$, including counts, `n+1` buckets, and collected values.
Hashing supports the expected-time qualification.
The input is unchanged.

### Failure Check

Do not replace frequency priority with numeric rank.
Within a bucket, this Python reference follows first appearance, not numeric order.
For `[2,1,3,3]`, `k=2` returns `[3,2]`; `[3,1]` is also a valid tied selection.
`k=0` returns `[]`.
Larger `k` is outside the teaching contract; the literal source returns all candidates rather than raising.

<!-- learning:core-end -->

## Change the Coordinate Storage

**New task:** Avoid allocating `n+1` buckets.

**Picture and change:** Keep exact counts, then select across the `u` distinct candidates with standard heap selection.
Retain `Counter` from the reference above.

```python
import heapq

def top_k_by_heap(nums, k):
   frequencies = Counter(nums)
   return heapq.nlargest(k, frequencies, key=frequencies.get)
```

**Check:** The source returns `[1,2]`; `k=0` returns `[]`.
Expected work is $O(n+u\log(k+1))$, with $O(u+k)$ state.
The exact count map remains: this is not an $O(k)$-memory heavy-hitter estimator.

**Use when:** The bucket range is costly relative to the distinct candidates and requested selection.

## Change the Tie Contract

**New task:** Equal frequencies must prefer the smaller numeric value.

**Picture and change:** Order distinct values by `(-frequency, value)`.
The extra key supplies an exact output order instead of an arbitrary cutoff choice.

```python
def top_k_deterministic(nums, k):
   frequencies = Counter(nums)
   return sorted(frequencies, key=lambda value: (-frequencies[value], value))[:k]
```

**Check:** `[2,1,3,3]`, `k=2`, now returns `[3,1]`.
Expected counting plus comparison ranking costs $O(n+u\log(u+1))$ time and $O(u)$ storage.
Do not keep the bucket method's linear ranking claim for this version.

**Use when:** Exact numeric tie order is part of the result contract.

### Transfer Check

Why not apply the same direct bucket scan to arbitrary real-valued scores?

### Answer

The small integer coordinate range justified the array allocation and scan.
Continuous or unbounded scores need another ordering structure, or an explicitly lossy quantization rule.
Compare [bounded value heaps](/questions/kth-largest-element/) when the weakest retained candidate should be exposed directly.
