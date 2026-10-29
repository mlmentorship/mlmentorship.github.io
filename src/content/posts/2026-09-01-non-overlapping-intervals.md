---
title: "Keep the earliest finishing compatible range to preserve future room"
description: "Prove maximum compatible cardinality by exchange, then request a witness or change to weighted interval selection."
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

Remove the fewest intervals needed so that all remaining intervals are compatible in time.

## Problem

**Input.** A list `intervals` of numeric pairs `[start, end]`, each satisfying `start < end`.

**Output.** Return the smallest number of intervals to remove so that no remaining intervals overlap.

**Rules.** Touching is compatible: one interval may start exactly when another ends.
Count removed occurrences, including repeated intervals, rather than removed duration.
Return only the removal count. The reference may reorder the input; an empty list returns zero.

**Example.** `[[1, 3], [3, 5], [2, 4], [5, 7]]` returns `1`.
Removing `[2, 4]` leaves three compatible intervals, including two pairs with touching endpoints.

## Picture

Sort by finish: `[1,3]`, `[2,4]`, `[3,5]`, `[5,7]`.
Keeping `[1,3]` leaves `[3,5]` available.
Choosing `[2,4]` first would block `[3,5]` without enabling any later choice unavailable after finish `3`.

Take an optimal nonempty schedule.
Replace its first interval with the earliest-finishing candidate.
Every later start still reaches that earlier finish, so the replacement preserves the schedule's count.
Repeat on the compatible suffix; an optimal solution can therefore follow every greedy choice.

Minimum removals equal total intervals minus maximum compatible cardinality.
Unlike union, this objective may discard covered time.

<!-- visual:non-overlapping-intervals-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="non-overlapping-intervals-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="non-overlapping-intervals-state-title">The earlier finishing interval leaves at least as much room for later choices</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="non-overlapping-intervals" role="group" tabindex="0" aria-label="Non-overlapping Intervals: Sort by end time; keep a candidate exactly when its start reaches the last kept end."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the fewest ranges to remove so the rest do not overlap.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize-end-order" role="group" aria-label="Initialize after sorting"><div class="coding-trace-frame-heading"><span>Initialize after sorting</span><strong>Sorting by end gives [1,3], [2,4], [3,5], [5,7]. Start with last_end=-inf and removed=0.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-1-3"><span>[1,3]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:14.285714285714285%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="" style="--trace-start:28.57142857142857%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-3-5"><span>[3,5]</span><div class="coding-trace-track"><i class="" style="--trace-start:42.857142857142854%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-5-7"><span>[5,7]</span><div class="coding-trace-track"><i class="" style="--trace-start:71.42857142857143%;--trace-width:28.57142857142857%"></i></div></div></div><div class="coding-trace-meta"><span><b>example</b>intervals = [[1,3], [3,5], [2,4], [5,7]]</span><span><b>sortedByEnd</b>[1,3], [2,4], [3,5], [5,7]</span><span><b>lastEnd</b>-inf</span><span><b>removed</b>0</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>max</dt><dd><strong class="coding-state-after">7</strong></dd></div><div class="coding-state-change"><dt>example</dt><dd><strong class="coding-state-after">intervals = [[1,3], [3,5], [2,4], [5,7]]</strong></dd></div><div class="coding-state-change"><dt>sorted by end</dt><dd><strong class="coding-state-after">[1,3], [2,4], [3,5], [5,7]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="keep-one-three" hidden role="group" aria-label="Keep the first interval"><div class="coding-trace-frame-heading"><span>Keep the first interval</span><strong>For [1,3], 1 &lt; -inf is false. Keep it and change last_end from -inf to 3.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-1-3"><span>[1,3]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:14.285714285714285%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="" style="--trace-start:28.57142857142857%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-3-5"><span>[3,5]</span><div class="coding-trace-track"><i class="" style="--trace-start:42.857142857142854%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-5-7"><span>[5,7]</span><div class="coding-trace-track"><i class="" style="--trace-start:71.42857142857143%;--trace-width:28.57142857142857%"></i></div></div></div><div class="coding-trace-meta"><span><b>example</b>intervals = [[1,3], [3,5], [2,4], [5,7]]</span><span><b>sortedByEnd</b>[1,3], [2,4], [3,5], [5,7]</span><span><b>comparison</b>1 &lt; -inf: false</span><span><b>lastEnd</b>3</span><span><b>removed</b>0</span><span><b>decision</b>keep [1,3]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>comparison</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1 &lt; -inf: false</strong></dd></div><div class="coding-state-change"><dt>last end</dt><dd><span class="coding-state-before">-inf</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>decision</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">keep [1,3]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="remove-two-four" hidden role="group" aria-label="Remove the overlap"><div class="coding-trace-frame-heading"><span>Remove the overlap</span><strong>For [2,4], 2 &lt; last_end 3 is true. Remove it, increment removed to 1, and leave last_end at 3.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-1-3"><span>[1,3]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:14.285714285714285%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:28.57142857142857%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-3-5"><span>[3,5]</span><div class="coding-trace-track"><i class="" style="--trace-start:42.857142857142854%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-5-7"><span>[5,7]</span><div class="coding-trace-track"><i class="" style="--trace-start:71.42857142857143%;--trace-width:28.57142857142857%"></i></div></div></div><div class="coding-trace-meta"><span><b>example</b>intervals = [[1,3], [3,5], [2,4], [5,7]]</span><span><b>sortedByEnd</b>[1,3], [2,4], [3,5], [5,7]</span><span><b>comparison</b>2 &lt; 3: true</span><span><b>lastEnd</b>3</span><span><b>removed</b>1</span><span><b>decision</b>remove [2,4]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>comparison</dt><dd><span class="coding-state-before">1 &lt; -inf: false</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2 &lt; 3: true</strong></dd></div><div class="coding-state-change"><dt>removed</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>decision</dt><dd><span class="coding-state-before">keep [1,3]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">remove [2,4]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="keep-three-five" hidden role="group" aria-label="Keep a touching interval"><div class="coding-trace-frame-heading"><span>Keep a touching interval</span><strong>For [3,5], 3 &lt; last_end 3 is false. Touching at 3 is allowed, so keep it and set last_end=5.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-1-3"><span>[1,3]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:14.285714285714285%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:28.57142857142857%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-3-5"><span>[3,5]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:42.857142857142854%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-5-7"><span>[5,7]</span><div class="coding-trace-track"><i class="" style="--trace-start:71.42857142857143%;--trace-width:28.57142857142857%"></i></div></div></div><div class="coding-trace-meta"><span><b>example</b>intervals = [[1,3], [3,5], [2,4], [5,7]]</span><span><b>sortedByEnd</b>[1,3], [2,4], [3,5], [5,7]</span><span><b>comparison</b>3 &lt; 3: false</span><span><b>lastEnd</b>5</span><span><b>removed</b>1</span><span><b>decision</b>keep [3,5]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>comparison</dt><dd><span class="coding-state-before">2 &lt; 3: true</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3 &lt; 3: false</strong></dd></div><div class="coding-state-change"><dt>last end</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5</strong></dd></div><div class="coding-state-change"><dt>decision</dt><dd><span class="coding-state-before">remove [2,4]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">keep [3,5]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="keep-five-seven" hidden role="group" aria-label="Keep the final interval"><div class="coding-trace-frame-heading"><span>Keep the final interval</span><strong>For [5,7], 5 &lt; last_end 5 is false. Keep it and set last_end=7.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-1-3"><span>[1,3]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:14.285714285714285%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:28.57142857142857%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-3-5"><span>[3,5]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:42.857142857142854%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-5-7"><span>[5,7]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:71.42857142857143%;--trace-width:28.57142857142857%"></i></div></div></div><div class="coding-trace-meta"><span><b>example</b>intervals = [[1,3], [3,5], [2,4], [5,7]]</span><span><b>sortedByEnd</b>[1,3], [2,4], [3,5], [5,7]</span><span><b>comparison</b>5 &lt; 5: false</span><span><b>lastEnd</b>7</span><span><b>removed</b>1</span><span><b>decision</b>keep [5,7]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>comparison</dt><dd><span class="coding-state-before">3 &lt; 3: false</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5 &lt; 5: false</strong></dd></div><div class="coding-state-change"><dt>last end</dt><dd><span class="coding-state-before">5</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">7</strong></dd></div><div class="coding-state-change"><dt>decision</dt><dd><span class="coding-state-before">keep [3,5]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">keep [5,7]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="return-one-removal" hidden role="group" aria-label="Return the removal count"><div class="coding-trace-frame-heading"><span>Return the removal count</span><strong>Exactly [2,4] was removed; [1,3], [3,5], and [5,7] remain pairwise non-overlapping.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-1-3"><span>[1,3]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:14.285714285714285%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:28.57142857142857%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-3-5"><span>[3,5]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:42.857142857142854%;--trace-width:28.57142857142857%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-5-7"><span>[5,7]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:71.42857142857143%;--trace-width:28.57142857142857%"></i></div></div></div><div class="coding-trace-meta"><span><b>example</b>intervals = [[1,3], [3,5], [2,4], [5,7]]</span><span><b>sortedByEnd</b>[1,3], [2,4], [3,5], [5,7]</span><span><b>lastEnd</b>7</span><span><b>kept</b>3</span><span><b>result</b>1</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>kept</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>comparison</dt><dd><span class="coding-state-before">5 &lt; 5: false</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize after sorting</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Keep the first interval</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Remove the overlap</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Keep a touching interval</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Keep the final interval</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Return the removal count</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">The task asks for the fewest removals needed to leave non-overlapping ranges, which is equivalent to keeping the largest compatible subset.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After each candidate, the kept intervals do not overlap, last_end is the end of the latest kept interval, and the greedy choices leave at least as much future room as any alternative of the same size.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When selecting the most compatible time ranges, an earlier finishing accepted choice never blocks a range that a later-finishing choice could keep; reuse this exchange argument for activity and reservation scheduling.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Sorting by end gives [1,3], [2,4], [3,5], [5,7]. Start with last_end=-inf and removed=0. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="non-overlapping-intervals-state"><strong>Check:</strong> Why is keeping [1,3] at least as good as keeping the overlapping [2,4] for all later choices?</span></figcaption></figure>

Keep `[1,3]`, reject `[2,4]`, then keep touching `[3,5]` and `[5,7]`.
`last_end` changes only after acceptance.
Rejecting an overlapping range must not move that boundary.

## Code

```python
def erase_overlap_intervals(intervals: list[list[int]]) -> int:
   removed = 0
   last_end = float("-inf")

   for start, end in sorted(intervals, key=lambda interval: interval[1]):
      if start < last_end:
         removed += 1
      else:
         last_end = end
   return removed
```

The scan retains $O(1)$ scalars, but Python's sorted list and sorting workspace use $O(n)$ storage.
Whole-function time is $O(n\log n)$; input rows remain unchanged.
Empty input returns `0`.

### Failure Check

Start-order selection can keep `[0,10]` while losing three compatible short ranges `[1,2]`, `[2,3]`, `[3,4]`.
Updating `last_end` after rejecting `[2,4]` would incorrectly block `[3,5]`.
The exchange proves cardinality, not maximum total weight.

<!-- learning:core-end -->

## Change to Witness Output

**New task:** Return a maximum compatible subset.

**Picture and change:** Append a copied range at each accepted transition.
The eligibility test and exchange proof stay unchanged.

```python
def compatible_intervals(intervals):
   kept, last_end = [], float("-inf")
   for start, end in sorted(intervals, key=lambda interval: interval[1]):
      if start >= last_end:
         kept.append([start, end])
         last_end = end
   return kept
```

**Check:** The source example returns `[[1,3],[3,5],[5,7]]`.
Its count must equal input length minus reference removals.
Adjacent chosen ranges must satisfy `previous_end <= next_start`.
Output adds storage proportional to the kept count; ties need not produce a unique witness.

**Use when:** The choices themselves must be shown.

## Change to Unequal Values

**New task:** Maximize total weight of compatible `(start,end,weight)` ranges.
Choosing nothing remains allowed.

**Picture and change:** End-sort and find each interval's last compatible prefix.
Compare skipping it with its weight plus that prefix optimum.
This uses [compatible prefix optima](/questions/house-robber/), not the equal-cardinality theorem.

```python
from bisect import bisect_right

def maximum_interval_weight(intervals):
   ordered = sorted(intervals, key=lambda interval: interval[1])
   ends = [end for start, end, weight in ordered]
   best = [0] * (len(ordered) + 1)
   for index, (start, end, weight) in enumerate(ordered):
      compatible = bisect_right(ends, start, 0, index)
      best[index + 1] = max(best[index], best[compatible] + weight)
   return best[-1]
```

**Check:** `[[0,3,100],[0,1,1],[1,2,1],[2,3,1]]` returns `100`.
Three weight-one intervals lose to one weight-100 interval despite their larger count.
Binary search includes earlier ends equal to the current start, preserving touching compatibility.
The weighted function uses $O(n\log n)$ scalar work and $O(n)$ storage, returning value rather than indices.

**Use when:** Interval weights replace equal-count value.

### Transfer Check

Why does union extend a maximum end while this algorithm prefers an earlier finish?

### Answer

Union preserves all covered points.
Scheduling may discard coverage to leave room for more choices.
The objectives justify different state updates.
