---
title: "Meeting Rooms: Make Every Conflict an Adjacent Boundary"
description: "Sort positive-length meetings, check neighboring boundaries, then adapt the endpoint and turnaround rules."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Mixed"
priority: "Core"
aliases: []
prerequisites: []
---

Determine whether one person can attend every meeting without overlapping commitments in the schedule.

## Problem

**Input.** A list `intervals` of integer meeting pairs `[start, end]`, with `start < end`.

**Output.** Return `True` if one person can attend all meetings in full, and `False` if any meetings conflict.

**Rules.** Meetings occupy half-open intervals `[start, end)`.
Ending at the next meeting's start is allowed; duplicate positive-duration meetings conflict.
The reference may reorder the outer list but does not alter its rows.
An empty schedule returns `True`. Zero-duration meetings are outside the contract.

**Example.** `[[0, 30], [5, 10], [15, 20]]` returns `False` because the first meeting overlaps the others.
For `[[0, 5], [5, 10]]`, return `True`.

## Picture

For `[[7, 10], [2, 4], [12, 14], [9, 11]]`, chronological order becomes `[2,4], [7,10], [9,11], [12,14]`.
`4 <= 7` passes; `10 <= 9` fails, proving overlap during `[9,10)`.
Skip the final boundary.

<!-- visual:meeting-rooms-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="meeting-rooms-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="meeting-rooms-state-title">One overlapping adjacent pair is enough to reject attendance</p><div class="visual-argument" data-visual-argument="meeting-rooms-state" data-argument-kind="matrix"><p class="visual-argument-premise">Worked example: touching endpoints are not overlapping meeting interiors</p><svg viewBox="0 0 360 160" role="img" aria-label="Worked example: touching endpoints are not overlapping meeting interiors. The test is next_start&lt;previous_end. Equality permits attendance; a strictly earlier start creates a conflict."><title>Worked example: touching endpoints are not overlapping meeting interiors</title><desc>The test is next_start&lt;previous_end. Equality permits attendance; a strictly earlier start creates a conflict.</desc><text x="126.48333333333335" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="126.48333333333335" dy="0">prev end</tspan></text><text x="211.89000000000001" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="211.89000000000001" dy="0">next</tspan><tspan x="211.89000000000001" dy="18">start</tspan></text><text x="297.29666666666674" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="297.29666666666674" dy="0">attend</tspan></text><text x="71.78" y="72" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">touch</tspan></text><rect data-proof-cell="0,0" x="83.78" y="54" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="124.98333333333335" y="81.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="124.98333333333335" dy="0">10</tspan></text><rect data-proof-cell="0,1" x="169.18666666666667" y="54" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="210.39000000000001" y="81.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="210.39000000000001" dy="0">10</tspan></text><rect data-proof-cell="0,2" x="254.59333333333336" y="54" width="82.40666666666668" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="295.7966666666667" y="81.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="295.7966666666667" dy="0">yes</tspan></text><text x="71.78" y="119" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">overlap</tspan></text><rect data-proof-cell="1,0" x="83.78" y="101" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="124.98333333333335" y="128.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="124.98333333333335" dy="0">10</tspan></text><rect data-proof-cell="1,1" x="169.18666666666667" y="101" width="82.40666666666668" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="210.39000000000001" y="128.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="210.39000000000001" dy="0">9</tspan></text><rect data-proof-cell="1,2" x="254.59333333333336" y="101" width="82.40666666666668" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="295.7966666666667" y="128.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="295.7966666666667" dy="0">NO</tspan></text><path d="M258.59333333333336 141L333.00000000000006 105" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="meeting-rooms" role="group" tabindex="0" aria-label="Meeting Rooms: Make Every Conflict an Adjacent Boundary: Sorting by start time makes the first overlap appear between adjacent meetings, so Python all can stop at the first failed boundary check."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether one person can attend every meeting.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="unsorted-intervals" role="group" aria-label="Start with unsorted meetings"><div class="coding-trace-frame-heading"><span>Start with unsorted meetings</span><strong>Use intervals [[7,10],[2,4],[12,14],[9,11]]. Their input order does not reveal the chronological neighbor checks.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-7-10"><span>[7,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:50%;--trace-width:21.428571428571427%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="" style="--trace-start:14.285714285714285%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-12-14"><span>[12,14]</span><div class="coding-trace-track"><i class="" style="--trace-start:85.71428571428571%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-9-11"><span>[9,11]</span><div class="coding-trace-track"><i class="" style="--trace-start:64.28571428571429%;--trace-width:14.285714285714285%"></i></div></div></div><div class="coding-trace-meta"><span><b>input</b>[[7,10],[2,4],[12,14],[9,11]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>max</dt><dd><strong class="coding-state-after">14</strong></dd></div><div class="coding-state-change"><dt>input</dt><dd><strong class="coding-state-after">[[7,10],[2,4],[12,14],[9,11]]</strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><strong class="coding-state-after">[7,10]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="sort-by-start" hidden role="group" aria-label="Sort lexicographically"><div class="coding-trace-frame-heading"><span>Sort lexicographically</span><strong>intervals.sort() orders starts as [[2,4],[7,10],[9,11],[12,14]]. Equal starts would be ordered by end, but this input has distinct starts.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="" style="--trace-start:14.285714285714285%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-7-10"><span>[7,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:50%;--trace-width:21.428571428571427%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-9-11"><span>[9,11]</span><div class="coding-trace-track"><i class="" style="--trace-start:64.28571428571429%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-12-14"><span>[12,14]</span><div class="coding-trace-track"><i class="" style="--trace-start:85.71428571428571%;--trace-width:14.285714285714285%"></i></div></div></div><div class="coding-trace-meta"><span><b>sorted</b>[[2,4],[7,10],[9,11],[12,14]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>sorted</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[[2,4],[7,10],[9,11],[12,14]]</strong></dd></div><div class="coding-state-change"><dt>input</dt><dd><span class="coding-state-before">[[7,10],[2,4],[12,14],[9,11]]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><span class="coding-state-before">[7,10]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[2,4]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="check-index-1" hidden role="group" aria-label="Check boundary at index 1"><div class="coding-trace-frame-heading"><span>Check boundary at index 1</span><strong>Compare previous end 4 with current start 7. Since 4 &lt;= 7, [2,4] finishes before [7,10] begins, and all continues.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:14.285714285714285%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-7-10"><span>[7,10]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:50%;--trace-width:21.428571428571427%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-9-11"><span>[9,11]</span><div class="coding-trace-track"><i class="" style="--trace-start:64.28571428571429%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-12-14"><span>[12,14]</span><div class="coding-trace-track"><i class="" style="--trace-start:85.71428571428571%;--trace-width:14.285714285714285%"></i></div></div></div><div class="coding-trace-meta"><span><b>index</b>1</span><span><b>boundary</b>4 &lt;= 7: true</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>index</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>boundary</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4 &lt;= 7: true</strong></dd></div><div class="coding-state-change"><dt>sorted</dt><dd><span class="coding-state-before">[[2,4],[7,10],[9,11],[12,14]]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="check-index-2" hidden role="group" aria-label="Check boundary at index 2"><div class="coding-trace-frame-heading"><span>Check boundary at index 2</span><strong>Compare previous end 10 with current start 9. The test 10 &lt;= 9 is false, and the bars overlap from time 9 to 10.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="" style="--trace-start:14.285714285714285%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-7-10"><span>[7,10]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:50%;--trace-width:21.428571428571427%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-9-11"><span>[9,11]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:64.28571428571429%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-12-14"><span>[12,14]</span><div class="coding-trace-track"><i class="" style="--trace-start:85.71428571428571%;--trace-width:14.285714285714285%"></i></div></div></div><div class="coding-trace-meta"><span><b>index</b>2</span><span><b>boundary</b>10 &lt;= 9: false</span><span><b>overlap</b>[9,10)</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>index</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>boundary</dt><dd><span class="coding-state-before">4 &lt;= 7: true</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">10 &lt;= 9: false</strong></dd></div><div class="coding-state-change"><dt>overlap</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[9,10)</strong></dd></div></dl><div class="coding-numeric-transition"><svg viewBox="0 0 360 158" role="img" aria-label="index: 1 before, 2 at the next shown checkpoint." style="display:block;width:100%;max-width:34rem;margin:auto"><title>index: 1 before, 2 at the next shown checkpoint.</title><desc>Each row has its own zero-inclusive numerical scale. Circles mark the preceding authored checkpoint and squares the current checkpoint. Connecting segments compare these shown values, not an invented path of execution.</desc><text x="16" y="20" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text-soft)">SHOWN CHECKPOINTS</text><text x="344" y="20" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text-soft)">circle: before / square: now</text><g data-numeric-transition="state:index" data-before="1" data-after="2" data-minimum="0" data-maximum="2"><text x="16" y="45" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">index</text><text x="344" y="45" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">1 -&gt; 2</text><path d="M28 96H332" style="stroke:var(--c-rule);fill:none"></path><path d="M28 65V111" style="stroke:var(--c-text-soft);stroke-dasharray:3 3;fill:none"></path><path d="M180 69L332 96" style="stroke:var(--viz-focus-stroke);stroke-width:2;fill:none"></path><circle data-checkpoint-before cx="180" cy="69" r="5" style="fill:var(--c-bg);stroke:var(--c-text-soft);stroke-width:2"></circle><rect data-checkpoint-after x="327" y="91" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="28" y="128" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">0</text><text x="332" y="128" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">2</text></g></svg></div></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="return-false" hidden role="group" aria-label="Short-circuit and return false"><div class="coding-trace-frame-heading"><span>Short-circuit and return false</span><strong>Python all stops after the false index-2 comparison, so index 3 is not evaluated. One person cannot attend [7,10] and [9,11].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row" data-motion-key="interval-2-4"><span>[2,4]</span><div class="coding-trace-track"><i class="" style="--trace-start:14.285714285714285%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-7-10"><span>[7,10]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:50%;--trace-width:21.428571428571427%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-9-11"><span>[9,11]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:64.28571428571429%;--trace-width:14.285714285714285%"></i></div></div><div class="coding-trace-interval-row" data-motion-key="interval-12-14"><span>[12,14]</span><div class="coding-trace-track"><i class="" style="--trace-start:85.71428571428571%;--trace-width:14.285714285714285%"></i></div></div></div><div class="coding-trace-meta"><span><b>skipped</b>index 3: 11 &lt;= 12 not evaluated</span><span><b>result</b>false</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>skipped</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">index 3: 11 &lt;= 12 not evaluated</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">false</strong></dd></div><div class="coding-state-change"><dt>index</dt><dd><span class="coding-state-before">2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start with unsorted meetings</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Sort lexicographically</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Check boundary at index 1</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Check boundary at index 2</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Short-circuit and return false</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when one resource must handle every interval and any temporal overlap makes the schedule invalid.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before checking index i, all adjacent boundaries before i are non-overlapping. In start-sorted order, any overlap must include an adjacent pair, so the first failed comparison proves impossibility.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Sort to make a global conflict local, then compare neighboring boundaries and stop at the first violation. This transfers to overlap validation, booking conflicts, and disjoint-range checks.</dd></div></dl></div><div class="visual-derivation-caption">Use intervals [[7,10],[2,4],[12,14],[9,11]]. Their input order does not reveal the chronological neighbor checks. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="meeting-rooms-state"><strong>Check:</strong> Which adjacent pair proves attendance is impossible? Why is the final pair never evaluated?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="meeting-rooms-state">The test is next_start&lt;previous_end. Equality permits attendance; a strictly earlier start creates a conflict.</span></figcaption></figure>

Why are neighbors sufficient?
Passing adjacent boundaries chain together: every earlier end precedes every later start.
A failed comparison exhibits a conflicting pair.

## Code

`all` stops at the first failed comparison.

```python
def can_attend_meetings(intervals: list[list[int]]) -> bool:
   intervals.sort()
   return all(
      intervals[index - 1][1] <= intervals[index][0]
      for index in range(1, len(intervals))
   )
```

**Cost:** $O(n\log n)$ comparisons and $O(n)$ sorting workspace; the scan needs constant auxiliary state.
Integer comparisons count as constant-cost operations.

## Change the Requirement

### Require Turnaround Time

**New task:** Require a nonnegative integer turnaround `gap` without input mutation.

**Picture and change:** Move each previous availability boundary from `end` to `end + gap`.

**Code:** Retain positive-length intervals; sort a copy.

```python
def can_attend_with_gap(intervals: list[list[int]], gap: int) -> bool:
   if gap < 0:
      raise ValueError("gap must be nonnegative")
   ordered = sorted(intervals)
   return all(
      ordered[index - 1][1] + gap <= ordered[index][0]
      for index in range(1, len(ordered))
   )
```

**Check:** `can_attend_with_gap([[1, 3], [4, 6]], 1)` returns `True`; gap `2` returns `False`.

**Use when:** Meetings require travel or reset time.
The availability boundaries still chain.

### Include Both Endpoints

**New task:** Include both endpoints.

**Picture and change:** Shared endpoints conflict; replace `<=` with `<`.

**Code:** Retain positive-length integers; avoid mutation.

```python
def can_attend_closed(intervals: list[list[int]]) -> bool:
   ordered = sorted(intervals)
   return all(
      ordered[index - 1][1] < ordered[index][0]
      for index in range(1, len(ordered))
   )
```

**Check:** `can_attend_closed([[1, 3], [3, 5]])` returns `False`; the original half-open call returns `True`.

**Use when:** Occupancy includes both endpoint instants.
Costs remain unchanged.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why exclude zero-duration rows from the half-open contract?

<details>
<summary>Reasoned answer</summary>

`[[1,4], [2,2]]` contains an empty half-open interval, yet the adjacent scan returns `False`.
If empty meetings should occupy no time, filter them before sorting; do not silently reuse the positive-length proof.
Sorting is also necessary: unsorted `[[5,6], [1,2]]` is compatible despite its descending input order.

</details>

</div>
