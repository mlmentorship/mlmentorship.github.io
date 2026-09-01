---
title: "Non-overlapping Intervals"
description: "Find the fewest ranges to remove so the rest do not overlap."
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

> Find the fewest ranges to remove so the rest do not overlap.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:non-overlapping-intervals-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="non-overlapping-intervals-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="non-overlapping-intervals-state-title">Non-overlapping Intervals: When intervals overlap, keep the one with the earlier end.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="non-overlapping-intervals" role="group" aria-label="Non-overlapping Intervals: When intervals overlap, keep the one with the earlier end."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the fewest ranges to remove so the rest do not overlap.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Sort by end"><div class="coding-trace-frame-heading"><span>Sort by end</span><strong>The candidate ending at 2 leaves the most room for later intervals.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,2]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:25%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row"><span>[1,3]</span><div class="coding-trace-track"><i class="" style="--trace-start:25%;--trace-width:50%"></i></div></div><div class="coding-trace-interval-row"><span>[2,3]</span><div class="coding-trace-track"><i class="" style="--trace-start:50%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row"><span>[3,4]</span><div class="coding-trace-track"><i class="" style="--trace-start:75%;--trace-width:25%"></i></div></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Reject the late-ending overlap"><div class="coding-trace-frame-heading"><span>Reject the late-ending overlap</span><strong>[1,3] overlaps the kept [1,2], so remove it and keep checking the remaining ranges.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,2]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:25%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row"><span>[1,3]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:25%;--trace-width:50%"></i></div></div><div class="coding-trace-interval-row"><span>[2,3]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:50%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row"><span>[3,4]</span><div class="coding-trace-track"><i class="" style="--trace-start:75%;--trace-width:25%"></i></div></div></div><div class="coding-trace-meta"><span><b>detail</b>remove 1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Keep room for the future"><div class="coding-trace-frame-heading"><span>Keep room for the future</span><strong>[2,3] or [3,4] can follow the earliest-ending choice.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,2]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:25%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row"><span>[2,3]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:50%;--trace-width:25%"></i></div></div><div class="coding-trace-interval-row"><span>[3,4]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:75%;--trace-width:25%"></i></div></div></div><div class="coding-trace-meta"><span><b>result</b>remove 1 interval</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Sort by end</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Reject the late-ending overlap</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Keep room for the future</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>When intervals overlap, keep the one with the earlier end.</p></div><figcaption><strong>Read it this way:</strong> The candidate ending at 2 leaves the most room for later intervals. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Greedy interval scheduling.

**Simple idea:** Keep the range that ends first. It leaves the most room for future ranges.
After sorting by end, remove any range that starts before the last kept end.

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

**Cost:** $O(n\log n)$ time and $O(1)$ extra space, not counting sorting.
