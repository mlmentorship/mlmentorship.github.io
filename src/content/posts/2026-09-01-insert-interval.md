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
<figure class="learning-figure coding-visual-figure" aria-labelledby="insert-interval-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="insert-interval-state-title">Insert Interval: Copy intervals before the new range, merge overlaps, then copy the suffix.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="insert-interval" role="group" aria-label="Insert Interval: Copy intervals before the new range, merge overlaps, then copy the suffix."><div class="coding-visual-example"><span>Input and goal</span><strong>Insert one range into sorted, non-overlapping ranges and merge when needed.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Copy the prefix"><div class="coding-trace-frame-heading"><span>Copy the prefix</span><strong>With new interval [4,8], [1,2] ends before it and stays untouched.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,2]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:10%;--trace-width:10%"></i></div></div><div class="coding-trace-interval-row"><span>[3,5]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:30%;--trace-width:20%"></i></div></div><div class="coding-trace-interval-row"><span>new [4,8]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:40%;--trace-width:40%"></i></div></div><div class="coding-trace-interval-row"><span>[6,9]</span><div class="coding-trace-track"><i class="" style="--trace-start:60%;--trace-width:30%"></i></div></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Merge the overlap"><div class="coding-trace-frame-heading"><span>Merge the overlap</span><strong>[4,8] overlaps [3,5] and [6,9], producing [3,9].</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,2]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:10%;--trace-width:10%"></i></div></div><div class="coding-trace-interval-row"><span>[3,9]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:30%;--trace-width:60%"></i></div></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return the ordered result"><div class="coding-trace-frame-heading"><span>Return the ordered result</span><strong>The final answer keeps the prefix and the merged range.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[1,2]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:10%;--trace-width:10%"></i></div></div><div class="coding-trace-interval-row"><span>[3,9]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:30%;--trace-width:60%"></i></div></div></div><div class="coding-trace-meta"><span><b>result</b>[[1,2],[3,9]]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Copy the prefix</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Merge the overlap</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the ordered result</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Copy intervals before the new range, merge overlaps, then copy the suffix.</p></div><figcaption><strong>Read it this way:</strong> With new interval [4,8], [1,2] ends before it and stays untouched. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
