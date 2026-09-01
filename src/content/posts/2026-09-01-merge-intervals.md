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
<figure class="learning-figure coding-visual-figure" aria-labelledby="merge-intervals-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="merge-intervals-state-title">Merge Intervals: Sort by start and extend the last merged interval whenever ranges overlap.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="merge-intervals" role="group" tabindex="0" aria-label="Merge Intervals: Sort by start and extend the last merged interval whenever ranges overlap."><div class="coding-visual-example"><span>Input and goal</span><strong>Merge every pair of overlapping ranges.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Start with the first range"><div class="coding-trace-frame-heading"><span>Start with the first range</span><strong>The merged output begins with [1,3].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,3]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:10%;--trace-width:20%"></i></div></div><div class="coding-trace-interval-row"><span>[2,6]</span><div class="coding-trace-track"><i class="" style="--trace-start:20%;--trace-width:40%"></i></div></div><div class="coding-trace-interval-row"><span>[8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:80%;--trace-width:20%"></i></div></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Extend on overlap"><div class="coding-trace-frame-heading"><span>Extend on overlap</span><strong>Since 2 &lt;= 3, merge [1,3] and [2,6] into [1,6].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,6]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:10%;--trace-width:50%"></i></div></div><div class="coding-trace-interval-row"><span>[8,10]</span><div class="coding-trace-track"><i class="" style="--trace-start:80%;--trace-width:20%"></i></div></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Start a new range"><div class="coding-trace-frame-heading"><span>Start a new range</span><strong>The next interval starts after 6, so it stays separate.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,6]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:10%;--trace-width:50%"></i></div></div><div class="coding-trace-interval-row"><span>[8,10]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:80%;--trace-width:20%"></i></div></div></div><div class="coding-trace-meta"><span><b>result</b>[[1,6],[8,10]]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start with the first range</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Extend on overlap</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Start a new range</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Sort by start and extend the last merged interval whenever ranges overlap.</p></div><figcaption><strong>Read it this way:</strong> The merged output begins with [1,3]. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
