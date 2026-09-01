---
title: "Merge Intervals"
description: "Merge every pair of overlapping ranges."
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

> Merge every pair of overlapping ranges.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:merge-intervals-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="merge-intervals-state-title"><p class="visual-kicker">Time ranges on one line</p><p class="visual-title" id="merge-intervals-state-title">Merge Intervals: Sorting makes the next possible conflict visible</p><div class="coding-visual coding-visual--interval" data-coding-visual data-coding-mode="interval" data-coding-slug="merge-intervals" role="group" aria-label="Merge Intervals: [1,3] and [2,6] overlap -&gt; carry [1,6]. The saved boundary summarizes every interval that can still affect the next one."><div class="coding-visual-example"><span>Concrete trace</span><strong>[1,3] and [2,6] overlap -&gt; carry [1,6]</strong></div><div class="coding-visual-sketch coding-visual-sketch--interval"><div class="coding-sketch-timeline"><span class="coding-sketch-tick">time</span><span class="coding-sketch-bar coding-sketch-bar--state">kept range</span><span class="coding-sketch-bar coding-sketch-bar--active">next range</span></div><p class="coding-sketch-note">sort first; carry the boundary that preserves future room</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Order</span><strong>start or end</strong><small>Put ranges in the order the proof needs.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Compare</span><strong>current boundary</strong><small>Check the next range against the active boundary.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Decide</span><strong>overlap?</strong><small>Merge, remove, or allocate a room.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Advance</span><strong>last safe end</strong><small>Carry the boundary that preserves future room.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The saved boundary summarizes every interval that can still affect the next one.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> See intervals as occupied segments, not pairs of unrelated numbers. The sort order turns a global overlap question into a local boundary comparison. For this problem, hold onto the concrete trace: [1,3] and [2,6] overlap -&gt; carry [1,6].</figcaption></figure>

**Pattern:** Sort by start, then scan.

**Simple idea:** Compare each range with the last merged range. Overlap extends the last end.
No overlap starts a new result range.

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

**Cost:** $O(n\log n)$ time and $O(n)$ answer space.
