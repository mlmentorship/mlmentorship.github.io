---
title: "Sort starts to keep only one open union component"
description: "Preserve covered points with a sorted union frontier, then change touching policy or count discrete coverage."
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

Combine overlapping closed intervals into disjoint ranges covering exactly the same points as the original intervals.

## Problem

**Input.** A list `intervals` of closed integer ranges `[start, end]`, each with `start <= end`.

**Output.** Return start-sorted, disjoint intervals whose union equals the union of all input ranges.

**Rules.** Closed intervals sharing an endpoint overlap and must merge.
Containment and repeated intervals do not add separate output ranges.
The reference may reorder the input list; empty input returns `[]`.

**Example.** `[[8, 10], [1, 3], [2, 6], [15, 18]]` returns `[[1, 6], [8, 10], [15, 18]]`.
For `[[1, 2], [2, 3]]`, return `[[1, 3]]` because the ranges share point 2.

## Picture

Sort to `[1,3]`, `[2,6]`, `[8,10]`, `[15,18]`.
The first two overlap, extending the open component to `[1,6]`.
The gap before `8` closes that component permanently.
Every future start is at least `8`, so it cannot overlap that earlier `[1,6]`.
The next gap similarly closes `[8,10]` before `[15,18]`.

After each prefix, output components are ordered, disjoint and cover exactly the processed ranges.
Only the last component remains eligible for extension.

<!-- visual:merge-intervals-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="merge-intervals-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="merge-intervals-state-title">Sorted order limits the next overlap to the final merged component</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="merge-intervals" role="group" tabindex="0" aria-label="Sort starts to keep only one open union component: Sort ranges by start; compare each range with the last merged end, extending on overlap or appending on a gap."><div class="coding-visual-example"><span>Input and goal</span><strong>Merge every pair of overlapping ranges.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="sort" role="group" aria-label="Sort the concrete input"><div class="coding-trace-frame-heading"><span>Sort the concrete input</span><strong>Sort the four ranges by start. Only the final merged component can overlap the next range.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-1-1-3"><span>1: [1,3]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:5.555555555555555%;--trace-width:11.11111111111111%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-2-2-6"><span>2: [2,6]</span><div class="coding-trace-track"><i class="" style="--trace-start:11.11111111111111%;--trace-width:22.22222222222222%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-3-8-10"><span>3: [8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:44.44444444444444%;--trace-width:11.11111111111111%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-4-15-18"><span>4: [15,18]</span><div class="coding-trace-track"><i class="" style="--trace-start:83.33333333333334%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>order</b>sorted by start</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>max</dt><dd><strong class="coding-state-after">18</strong></dd></div><div class="coding-state-change"><dt>order</dt><dd><strong class="coding-state-after">sorted by start</strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><strong class="coding-state-after">1: [1,3]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="seed" hidden role="group" aria-label="Seed the merged output"><div class="coding-trace-frame-heading"><span>Seed the merged output</span><strong>Copy ordered[0] = [1,3]. The last merged end is now 3.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-merged-1-3"><span>merged [1,3]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:5.555555555555555%;--trace-width:11.11111111111111%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-next-2-6"><span>next [2,6]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:11.11111111111111%;--trace-width:22.22222222222222%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-pending-8-10"><span>pending [8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:44.44444444444444%;--trace-width:11.11111111111111%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-pending-15-18"><span>pending [15,18]</span><div class="coding-trace-track"><i class="" style="--trace-start:83.33333333333334%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>state</b>merged = [[1,3]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>state</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">merged = [[1,3]]</strong></dd></div><div class="coding-state-change"><dt>order</dt><dd><span class="coding-state-before">sorted by start</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><span class="coding-state-before">1: [1,3]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">merged [1,3]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="merge-overlap" hidden role="group" aria-label="Extend the overlapping end"><div class="coding-trace-frame-heading"><span>Extend the overlapping end</span><strong>For [2,6], start 2 &lt;= last end 3, so set the last end to max(3,6) = 6.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-merged-1-6"><span>merged [1,6]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:5.555555555555555%;--trace-width:27.77777777777778%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-current-2-6"><span>current [2,6]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:11.11111111111111%;--trace-width:22.22222222222222%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-next-8-10"><span>next [8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:44.44444444444444%;--trace-width:11.11111111111111%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-pending-15-18"><span>pending [15,18]</span><div class="coding-trace-track"><i class="" style="--trace-start:83.33333333333334%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>branch</b>2 &lt;= 3 -&gt; overlap</span><span><b>state</b>merged = [[1,6]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>branch</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2 &lt;= 3 -&gt; overlap</strong></dd></div><div class="coding-state-change"><dt>state</dt><dd><span class="coding-state-before">merged = [[1,3]]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">merged = [[1,6]]</strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><span class="coding-state-before">merged [1,3]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">merged [1,6]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="append-eight-ten" hidden role="group" aria-label="Append after the first gap"><div class="coding-trace-frame-heading"><span>Append after the first gap</span><strong>For [8,10], start 8 &gt; last end 6, so append a new merged range instead of extending [1,6].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-merged-1-6"><span>merged [1,6]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:5.555555555555555%;--trace-width:27.77777777777778%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-merged-8-10"><span>merged [8,10]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:44.44444444444444%;--trace-width:11.11111111111111%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-current-8-10"><span>current [8,10]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:44.44444444444444%;--trace-width:11.11111111111111%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-next-15-18"><span>next [15,18]</span><div class="coding-trace-track"><i class="" style="--trace-start:83.33333333333334%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>branch</b>8 &gt; 6 -&gt; append</span><span><b>state</b>merged = [[1,6],[8,10]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>branch</dt><dd><span class="coding-state-before">2 &lt;= 3 -&gt; overlap</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">8 &gt; 6 -&gt; append</strong></dd></div><div class="coding-state-change"><dt>state</dt><dd><span class="coding-state-before">merged = [[1,6]]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">merged = [[1,6],[8,10]]</strong></dd></div><div class="coding-state-change"><dt>index 1</dt><dd><span class="coding-state-before">current [2,6]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">merged [8,10]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="append-fifteen-eighteen" hidden role="group" aria-label="Append after the second gap"><div class="coding-trace-frame-heading"><span>Append after the second gap</span><strong>For [15,18], start 15 &gt; last end 10, so append it. Every sorted interval has now been consumed.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-merged-1-6"><span>merged [1,6]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:5.555555555555555%;--trace-width:27.77777777777778%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-merged-8-10"><span>merged [8,10]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:44.44444444444444%;--trace-width:11.11111111111111%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-merged-15-18"><span>merged [15,18]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:83.33333333333334%;--trace-width:16.666666666666664%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-current-15-18"><span>current [15,18]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:83.33333333333334%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>branch</b>15 &gt; 10 -&gt; append</span><span><b>result</b>[[1,6],[8,10],[15,18]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>branch</dt><dd><span class="coding-state-before">8 &gt; 6 -&gt; append</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">15 &gt; 10 -&gt; append</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[[1,6],[8,10],[15,18]]</strong></dd></div><div class="coding-state-change"><dt>state</dt><dd><span class="coding-state-before">merged = [[1,6],[8,10]]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Sort the concrete input</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Seed the merged output</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Extend the overlapping end</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Append after the first gap</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Append after the second gap</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when arbitrary ranges must be coalesced by overlap; sorting by start guarantees that any range able to overlap the current merged component appears before a later gap.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before each current interval, merged is sorted, disjoint, and exactly covers all processed ranges; only merged[-1] can overlap current because starts are nondecreasing.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Sorting can turn a global overlap problem into a local frontier check: preserve a completed prefix and keep only the final component open for possible extension.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Sort the four ranges by start. Only the final merged component can overlap the next range. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="merge-intervals-state"><strong>Check:</strong> Why can the next sorted interval affect only the final merged component?</span></figcaption></figure>

On overlap, retain the larger end.
For `[1,10]` followed by `[2,3]`, replacing the end with `3` would lose covered points.
On a gap, append a fresh component.

## Code

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
   if not intervals:
      return []

   ordered = sorted(intervals)
   merged = [ordered[0].copy()]

   for start, end in ordered[1:]:
      if start <= merged[-1][1]:
         merged[-1][1] = max(merged[-1][1], end)
      else:
         merged.append([start, end])
   return merged
```

`sorted` preserves input order and rows; the copied first row prevents later frontier updates from mutating an input row.
Every returned row is newly allocated.
Empty input returns `[]`.

Sorting takes $O(n\log n)$ time.
Answer storage is $O(n)$; sorted lists, slicing and sorting workspace also take $O(n)$ space.
This is union, not maximum compatible selection or peak simultaneous resource demand.

<!-- learning:core-end -->

## Change the Touching Policy

**New task:** Coalesce only positive-width overlaps, leaving merely touching ranges separate.

**Picture and change:** Use strict `start < last_end` and require positive-width inputs.
This changes grouping policy, not necessarily the covered real set.

```python
def merge_strict_overlaps(intervals):
   merged = []
   for start, end in sorted(intervals):
      if merged and start < merged[-1][1]:
         merged[-1][1] = max(merged[-1][1], end)
      else:
         merged.append([start, end])
   return merged
```

**Check:** `[[1,2],[2,3],[2,4]]` returns `[[1,2],[2,4]]`.
The first range only touches the second group; the equal-start ranges have positive-width overlap and merge.
Neither endpoint policy should be silently imported from a scheduling task.

**Use when:** Shared endpoints alone must not coalesce groups.

## Change to Integer Coverage

**New task:** Count distinct integer coordinates covered by closed integer ranges.

**Picture and change:** First obtain disjoint union components.
Each contributes `end-start+1` integer points without enumerating them.

```python
def covered_integer_points(intervals):
   return sum(end - start + 1
            for start, end in merge_intervals(intervals))
```

**Check:** `[[1,3],[2,6],[8,10]]` covers six points `1..6` and three points `8..10`: total `9`.
Continuous length would omit the `+1` and answer a different question.
Both adaptations retain the sorting bound and fresh union rows.

**Use when:** Union representation becomes a discrete coverage count.

### Transfer Check

Why cannot earliest-finish scheduling replace this scan?

### Answer

Scheduling discards overlapping ranges to maximize a compatible subset.
Union must preserve every covered point, so it extends components instead of rejecting their coverage.
