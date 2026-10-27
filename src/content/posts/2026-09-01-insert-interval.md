---
title: "Insert Interval"
description: "Insert one range into sorted, non-overlapping ranges and merge when needed."
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

Insert one interval into an ordered disjoint interval list and return the resulting merged coverage.

## Problem

**Input.** Start-sorted, pairwise disjoint closed integer `intervals` and one closed `new_interval = [start, end]`.

**Output.** Return a start-sorted disjoint list covering exactly the old intervals and the new interval together.

**Rules.** Every interval has `start <= end`. Overlapping intervals and intervals sharing an endpoint must merge.
Do not change either input. The reference may reuse untouched input rows in the output rather than copying them.
With no old intervals, return the new interval's values as the only interval.

**Example.** Inserting `[2, 5]` into `[[1, 3], [6, 9]]` returns `[[1, 5], [6, 9]]`.
Inserting `[3, 6]` into the same list returns `[[1, 9]]` because both endpoints connect.

## Picture

Inserting `[4,8]` into `[[1,2],[3,5],[6,7],[8,10],[12,16]]` keeps `[1,2]`, grows `[3,10]`, then reuses `[12,16]`.

<!-- visual:insert-interval-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="insert-interval-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="insert-interval-state-title">Insert Interval: Partition sorted ranges into a copied prefix, one growing overlap span, and an untouched suffix.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="insert-interval" role="group" tabindex="0" aria-label="Insert Interval: Partition sorted ranges into a copied prefix, one growing overlap span, and an untouched suffix."><div class="coding-visual-example"><span>Input and goal</span><strong>Insert one range into sorted, non-overlapping ranges and merge when needed.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize the inserted range"><div class="coding-trace-frame-heading"><span>Initialize the inserted range</span><strong>Use intervals [[1,2],[3,5],[6,7],[8,10],[12,16]] and new [4,8]. Start answer = [], index = 0, span = [4,8].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-current-1-2"><span>current [1,2]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:6.25%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-3-5"><span>[3,5]</span><div class="coding-trace-track"><i class="" style="--trace-start:18.75%;--trace-width:12.5%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-new-span-4-8"><span>new span [4,8]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:25%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-6-7"><span>[6,7]</span><div class="coding-trace-track"><i class="" style="--trace-start:37.5%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-8-10"><span>[8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:50%;--trace-width:12.5%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-12-16"><span>[12,16]</span><div class="coding-trace-track"><i class="" style="--trace-start:75%;--trace-width:25%"></i></div></div></div><div class="coding-trace-meta"><span><b>state</b>answer = []; span = [4,8]</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="copy-prefix" hidden role="group" aria-label="Copy the interval fully before"><div class="coding-trace-frame-heading"><span>Copy the interval fully before</span><strong>[1,2] ends at 2 &lt; span start 4, so append [1,2] unchanged and advance index to [3,5].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-answer-1-2"><span>answer [1,2]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:6.25%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-current-3-5"><span>current [3,5]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:18.75%;--trace-width:12.5%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-span-4-8"><span>span [4,8]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:25%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-pending-6-7"><span>pending [6,7]</span><div class="coding-trace-track"><i class="" style="--trace-start:37.5%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-pending-8-10"><span>pending [8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:50%;--trace-width:12.5%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-suffix-12-16"><span>suffix [12,16]</span><div class="coding-trace-track"><i class="" style="--trace-start:75%;--trace-width:25%"></i></div></div></div><div class="coding-trace-meta"><span><b>branch</b>2 &lt; 4 -&gt; copy prefix</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="merge-three-five" hidden role="group" aria-label="Merge the first overlap"><div class="coding-trace-frame-heading"><span>Merge the first overlap</span><strong>[3,5] starts at 3 &lt;= span end 8, so span becomes [min(4,3), max(8,5)] = [3,8].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-answer-1-2"><span>answer [1,2]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:6.25%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-current-3-5"><span>current [3,5]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:18.75%;--trace-width:12.5%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-growing-span-3-8"><span>growing span [3,8]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:18.75%;--trace-width:31.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-next-6-7"><span>next [6,7]</span><div class="coding-trace-track"><i class="" style="--trace-start:37.5%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-pending-8-10"><span>pending [8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:50%;--trace-width:12.5%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-suffix-12-16"><span>suffix [12,16]</span><div class="coding-trace-track"><i class="" style="--trace-start:75%;--trace-width:25%"></i></div></div></div><div class="coding-trace-meta"><span><b>arithmetic</b>[min(4,3), max(8,5)] = [3,8]</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="merge-six-seven" hidden role="group" aria-label="Consume a contained overlap"><div class="coding-trace-frame-heading"><span>Consume a contained overlap</span><strong>[6,7] starts at 6 &lt;= 8, but [min(3,6), max(8,7)] stays [3,8]. Advance without emitting a second range.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-answer-1-2"><span>answer [1,2]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:6.25%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-growing-span-3-8"><span>growing span [3,8]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:18.75%;--trace-width:31.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-current-6-7"><span>current [6,7]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:37.5%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-next-8-10"><span>next [8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:50%;--trace-width:12.5%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-suffix-12-16"><span>suffix [12,16]</span><div class="coding-trace-track"><i class="" style="--trace-start:75%;--trace-width:25%"></i></div></div></div><div class="coding-trace-meta"><span><b>arithmetic</b>[min(3,6), max(8,7)] = [3,8]</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="merge-eight-ten" hidden role="group" aria-label="Extend through a touching overlap"><div class="coding-trace-frame-heading"><span>Extend through a touching overlap</span><strong>[8,10] starts at 8 &lt;= span end 8, so closed intervals overlap at 8 and span extends to [3,10].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-answer-1-2"><span>answer [1,2]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:6.25%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-growing-span-3-10"><span>growing span [3,10]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:18.75%;--trace-width:43.75%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-current-8-10"><span>current [8,10]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:50%;--trace-width:12.5%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-next-12-16"><span>next [12,16]</span><div class="coding-trace-track"><i class="" style="--trace-start:75%;--trace-width:25%"></i></div></div></div><div class="coding-trace-meta"><span><b>arithmetic</b>[min(3,8), max(8,10)] = [3,10]</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="return-result" hidden role="group" aria-label="Stop at the suffix gap"><div class="coding-trace-frame-heading"><span>Stop at the suffix gap</span><strong>[12,16] starts at 12 &gt; span end 10, so overlap scanning stops. Return prefix + [3,10] + untouched suffix.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-answer-1-2"><span>answer [1,2]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:6.25%;--trace-width:6.25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-inserted-3-10"><span>inserted [3,10]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:18.75%;--trace-width:43.75%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-suffix-12-16"><span>suffix [12,16]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:75%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-current-12-16"><span>current [12,16]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:75%;--trace-width:25%"></i></div></div></div><div class="coding-trace-meta"><span><b>branch</b>12 &gt; 10 -&gt; suffix begins</span><span><b>result</b>[[1,2],[3,10],[12,16]]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize the inserted range</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Copy the interval fully before</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Merge the first overlap</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Consume a contained overlap</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Extend through a touching overlap</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Stop at the suffix gap</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when inserting one range into already sorted disjoint ranges; their order partitions them into intervals ending before, overlapping, or starting after the growing span.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">answer contains every interval strictly before the span, span is the union of the new interval and all consumed overlaps, and intervals from index onward remain untouched and sorted.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When ordered input surrounds one mutable object, scan an immutable prefix, absorb the contiguous interaction region into that object, then reuse the untouched suffix.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Use intervals [[1,2],[3,5],[6,7],[8,10],[12,16]] and new [4,8]. Start answer = [], index = 0, span = [4,8]. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

The prefix ends before `start`; every subsequent overlapping row can only expand `end`.
Once the next start exceeds `end`, sorted disjointness makes the entire suffix safe.
This is [interval union](/questions/merge-intervals/) without resorting already ordered input.

## Code

```python
def insert_interval(intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
   answer: list[list[int]] = []
   index = 0
   start, end = new_interval

   while index < len(intervals) and intervals[index][1] < start:
      answer.append(intervals[index])
      index += 1

   while index < len(intervals) and intervals[index][0] <= end:
      start = min(start, intervals[index][0])
      end = max(end, intervals[index][1])
      index += 1

   return answer + [[start, end]] + intervals[index:]
```

**Cost:** $O(n)$ time and $O(n)$ answer space.

Slicing and concatenation also allocate $O(n)$ temporary references.
Integer comparisons count as constant-cost operations.

## Change the Requirement

### Detach the Result

**New task:** Prevent result-row mutations from affecting either input.

**Picture and change:** Copy every row after computing the same union.

**Code:** Reuse the complete reference.

```python
def insert_interval_copy(intervals: list[list[int]], new_interval: list[int]):
   return [row[:] for row in insert_interval(intervals, new_interval)]
```

**Check:** Inserting `[5,6]` into `[[1,2]]` returns `[[1,2],[5,6]]` with fresh row objects.

**Use when:** Callers need independent ownership; time and storage remain linear.

### Exclude Right Endpoints

**New task:** Keep touching half-open intervals separate; require positive durations.

**Picture and change:** The prefix accepts `end <= start`; merging requires `next_start < end`.

**Code:** Existing rows must be sorted and nonoverlapping under the half-open convention.

```python
def insert_half_open(intervals: list[list[int]], new_interval: list[int]):
   answer, index = [], 0
   start, end = new_interval
   while index < len(intervals) and intervals[index][1] <= start:
      answer.append(intervals[index])
      index += 1
   while index < len(intervals) and intervals[index][0] < end:
      start = min(start, intervals[index][0])
      end = max(end, intervals[index][1])
      index += 1
   return answer + [[start, end]] + intervals[index:]
```

**Check:** `insert_half_open([[1,3]], [3,5])` returns `[[1,3],[3,5]]`; the closed reference returns `[[1,5]]`.

**Use when:** An endpoint touch is not a merging event.
The aliasing and linear bounds remain unchanged.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why not accept arbitrary unsorted rows?

<details>
<summary>Reasoned answer</summary>

With `[[8,10],[1,2]]` and `[3,4]`, the first row already looks like a safe suffix.
The reference never discovers the later earlier interval, so its result is not sorted.
Sort and merge arbitrary input first; the linear proof depends on the original ordering and disjointness.

</details>

</div>
