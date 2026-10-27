---
title: "Subarray Sum Equals K"
description: "Count continuous subarrays whose sum equals the target."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Foundation"
priority: "Core"
aliases: []
prerequisites: []
---

Count the nonempty contiguous ranges whose values add to a requested target sum.

## Problem

**Input.** An integer list `nums` and an integer `target`.

**Output.** Return the number of nonempty contiguous subarrays whose sum equals `target`.

**Rules.** Count ranges by their start and end positions; overlapping ranges and identical value sequences count separately.
Negative values, zeros, and duplicates are allowed. Do not count the empty range, even for target zero.
Leave the input unchanged. Empty input returns zero.

**Example.** `[1, -1, 1, -1, 1]` with target zero returns `6`.
The valid ranges have lengths two or four. For `[1, 1, 1]` and target 2, return `2`.

## Picture

Put a boundary before the array and after each element.
Write the total so far at each boundary: `0, 1, 0, 1, 0, 1`.
Two equal totals enclose a zero-sum range. Their positions determine its start and end.

<!-- visual:subarray-sum-equals-k-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="subarray-sum-equals-k-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="subarray-sum-equals-k-state-title">Subarray Sum Equals K: Count earlier prefix sums equal to current prefix minus target.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="subarray-sum-equals-k" role="group" tabindex="0" aria-label="Subarray Sum Equals K: Count earlier prefix sums equal to current prefix minus target."><div class="coding-visual-example"><span>Input and goal</span><strong>Count continuous subarrays whose sum equals the target.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="empty-prefix" role="group" aria-label="Seed the empty prefix"><div class="coding-trace-frame-heading"><span>Seed the empty prefix</span><strong>Use nums = [1, -1, 1, -1, 1] and target 0. Record the empty boundary once so ranges starting at index 0 can match.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="scan">next i=0</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>earlier prefix counts</span><span><b>prefix</b>0</span><span><b>target</b>0</span><span><b>answer</b>0</span></div><div class="coding-trace-map"><span class="coding-trace-label">earlier prefix counts</span><span class="coding-trace-map-entry"><b>0</b><span>count 1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="scan-index-0" hidden role="group" aria-label="Read index 0"><div class="coding-trace-frame-heading"><span>Read index 0</span><strong>Prefix becomes 1. Need 1 - 0 = 1, which has appeared 0 times; answer stays 0, then prefix 1 is stored once.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="scan">i=0</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>query</b>1 - 0 = 1; prior count 0</span><span><b>prefix</b>1</span><span><b>answer</b>0 + 0 = 0</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>0</b><span>count 1</span></span><span class="coding-trace-map-entry"><b>1</b><span>count 1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="scan-index-1" hidden role="group" aria-label="Read index 1"><div class="coding-trace-frame-heading"><span>Read index 1</span><strong>Prefix returns to 0. Need prefix 0, seen once, so [0..1] is counted; then the stored count for 0 becomes 2.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="scan">i=1</span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>query</b>0 - 0 = 0; prior count 1</span><span><b>matched</b>[0..1]</span><span><b>answer</b>0 + 1 = 1</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>0</b><span>count 2</span></span><span class="coding-trace-map-entry"><b>1</b><span>count 1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="scan-index-2" hidden role="group" aria-label="Read index 2"><div class="coding-trace-frame-heading"><span>Read index 2</span><strong>Prefix becomes 1. Need prefix 1, seen once, so [1..2] is counted; then the stored count for 1 becomes 2.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="scan">i=2</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>query</b>1 - 0 = 1; prior count 1</span><span><b>matched</b>[1..2]</span><span><b>answer</b>1 + 1 = 2</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>0</b><span>count 2</span></span><span class="coding-trace-map-entry"><b>1</b><span>count 2</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="scan-index-3" hidden role="group" aria-label="Read index 3"><div class="coding-trace-frame-heading"><span>Read index 3</span><strong>Prefix returns to 0. Two earlier zero prefixes produce [0..3] and [2..3]; add both before raising count(0) to 3.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="scan">i=3</span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>query</b>0 - 0 = 0; prior count 2</span><span><b>matched</b>[0..3], [2..3]</span><span><b>answer</b>2 + 2 = 4</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>0</b><span>count 3</span></span><span class="coding-trace-map-entry"><b>1</b><span>count 2</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="scan-index-4" hidden role="group" aria-label="Read index 4"><div class="coding-trace-frame-heading"><span>Read index 4</span><strong>Prefix becomes 1. Two earlier prefix-1 positions produce [1..4] and [3..4]; the final count is 6.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="scan">i=4</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>query</b>1 - 0 = 1; prior count 2</span><span><b>matched</b>[1..4], [3..4]</span><span><b>answer</b>4 + 2 = 6</span><span><b>result</b>6 subarrays</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>0</b><span>count 3</span></span><span class="coding-trace-map-entry"><b>1</b><span>count 3</span></span></div></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Seed the empty prefix</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Read index 0</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read index 1</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Read index 2</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Read index 3</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Read index 4</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">The task counts contiguous ranges with an exact sum, values may be negative, and many ranges can end at the same index, so a monotone sliding window is unsafe.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before processing nums[i], the map counts every prefix ending before i. After prefix is updated, count(prefix - target) is exactly the number of valid subarrays ending at i; only then is the current prefix inserted.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Rewrite a range equation as current prefix minus earlier prefix. Store counts when duplicate earlier states represent distinct starts; this also transfers to subarray sums divisible by k and equal-count prefix signatures.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Use nums = [1, -1, 1, -1, 1] and target 0. Record the empty boundary once so ranges starting at index 0 can match. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

At a current total `prefix`, a valid start has total `prefix - target`.
`prefix_count` stores how many earlier boundaries have each total.

| Element index | Current total | Earlier matches | New answer |
| --- | --- | --- | --- |
| 0 | 1 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| 2 | 1 | 1 | 2 |
| 3 | 0 | 2 | 4 |
| 4 | 1 | 2 | 6 |

The initial `{0: 1}` represents the boundary before index `0`.
Query before inserting the current boundary. Otherwise, target `0` would count an empty range.
Each range is counted once, at its right boundary.
Negative values do not change this subtraction rule.

## Code

```python
def subarray_sum(nums: list[int], target: int) -> int:
   prefix_count = {0: 1}
   prefix = 0
   answer = 0

   for num in nums:
      prefix += num
      answer += prefix_count.get(prefix - target, 0)
      prefix_count[prefix] = prefix_count.get(prefix, 0) + 1

   return answer
```

There are $n$ additions and expected constant-time map operations.
Time is expected $O(n)$ and extra space is $O(n)$ under bounded-size integer arithmetic.
Python integers do not overflow, but larger integers cost more to add and hash.

## Change what the map saves

### 1. Return the longest range

**New task:** return the maximum length, not the number of ranges.

**Picture and change:** for a fixed end, the earliest matching start gives the longest range.
Replace counts with earliest indices. Boundary `-1` lies before the first element.
The whole replacement function follows.

```python
def longest_sum(nums, target):
   earliest = {0: -1}
   prefix = best = 0
   for index, value in enumerate(nums):
      prefix += value
      if prefix - target in earliest:
         best = max(best, index - earliest[prefix - target])
      earliest.setdefault(prefix, index)
   return best
```

**Check:** `longest_sum([1, -1, 1, -1, 1], 0)` returns `4`.
At index `3`, total `0` matches boundary `-1`, giving length `4`.
Overwriting `earliest[0]` would lose that start.

**Use when:** the result asks for length. The boundary subtraction stays unchanged.

### 2. Return whether a range exists

**New task:** return a boolean for any nonempty matching range.

**Picture and change:** membership is enough. Replace counts with a set and return on the first match.
Still query before inserting. The whole replacement function follows.

```python
def has_sum(nums, target):
   seen = {0}
   prefix = 0
   for value in nums:
      prefix += value
      if prefix - target in seen:
         return True
      seen.add(prefix)
   return False
```

**Check:** `has_sum([], 0)` is `False`; `has_sum([1, -1], 0)` is `True`.
The second total matches the earlier boundary `0`.

**Use when:** only existence matters. Both alternatives retain expected $O(n)$ time and $O(n)$ extra space.

## Failure case

A set loses repeated starts, so it cannot count all ranges.
For `[0, 0]`, the answer is `3`, not `2`.
A sliding window also needs a different proof: later negative values can reduce an oversized sum.

## Try a changed result

Why is one saved index enough for the longest range but insufficient for the range count?

<details>
<summary>Check your answer</summary>

Counting needs every distinct start. For a fixed end, maximum length needs only the earliest valid start.

</details>

<!-- learning:core-end -->
