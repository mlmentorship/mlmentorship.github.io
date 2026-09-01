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

> Insert one range into sorted, non-overlapping ranges and merge when needed.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:insert-interval-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="insert-interval-state-title"><p class="visual-kicker">Time ranges on one line</p><p class="visual-title" id="insert-interval-state-title">Insert Interval: Sorting makes the next possible conflict visible</p><div class="coding-visual coding-visual--interval" data-coding-visual data-coding-mode="interval" data-coding-slug="insert-interval" role="group" aria-label="Insert Interval: before, overlap, after -&gt; copy [1,2], merge [3,5] with [4,8], copy [10,12]. The saved boundary summarizes every interval that can still affect the next one."><div class="coding-visual-example"><span>Concrete trace</span><strong>before, overlap, after -&gt; copy [1,2], merge [3,5] with [4,8], copy [10,12]</strong></div><div class="coding-visual-sketch coding-visual-sketch--interval"><div class="coding-sketch-timeline"><span class="coding-sketch-tick">time</span><span class="coding-sketch-bar coding-sketch-bar--state">kept range</span><span class="coding-sketch-bar coding-sketch-bar--active">next range</span></div><p class="coding-sketch-note">sort first; carry the boundary that preserves future room</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Order</span><strong>start or end</strong><small>Put ranges in the order the proof needs.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Compare</span><strong>current boundary</strong><small>Check the next range against the active boundary.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Decide</span><strong>overlap?</strong><small>Merge, remove, or allocate a room.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Advance</span><strong>last safe end</strong><small>Carry the boundary that preserves future room.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The saved boundary summarizes every interval that can still affect the next one.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> See intervals as occupied segments, not pairs of unrelated numbers. The sort order turns a global overlap question into a local boundary comparison. For this problem, hold onto the concrete trace: before, overlap, after -&gt; copy [1,2], merge [3,5] with [4,8], copy [10,12].</figcaption></figure>

**Pattern:** Three interval groups.

**Simple idea:** First copy ranges fully before the new range. Then merge every overlap.
Finally copy the ranges fully after it.

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
