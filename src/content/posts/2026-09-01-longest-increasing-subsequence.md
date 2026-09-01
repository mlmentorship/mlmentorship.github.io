---
title: "Longest Increasing Subsequence"
description: "Find the longest strictly increasing subsequence. Values do not need to be next to each other."
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

> Find the longest strictly increasing subsequence. Values do not need to be next to each other.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-increasing-subsequence-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-increasing-subsequence-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-increasing-subsequence-state-title">Longest Increasing Subsequence: For each subsequence length, keep the smallest possible ending value.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-increasing-subsequence" role="group" aria-label="Longest Increasing Subsequence: For each subsequence length, keep the smallest possible ending value."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the longest strictly increasing subsequence. Values do not need to be next to each other.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Read 10, 9, 2"><div class="coding-trace-frame-heading"><span>Read 10, 9, 2</span><strong>Each new smaller value replaces the tail for length 1.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">10</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">9</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">tails=[2]</span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">101</span></span></div><div class="coding-trace-meta"><span><b>tails</b>[2]</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Extend and replace tails"><div class="coding-trace-frame-heading"><span>Extend and replace tails</span><strong>5, 3, and 7 produce tails [2,3,7].</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">length 3</span><span class="coding-trace-array-cell">7</span></span></div><div class="coding-trace-meta"><span><b>tails</b>[2,3,7]</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Append 101"><div class="coding-trace-frame-heading"><span>Append 101</span><strong>101 extends the tail list, giving length 4.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">append</span><span class="coding-trace-array-cell">101</span></span></div><div class="coding-trace-meta"><span><b>result</b>4</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Read 10, 9, 2</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Extend and replace tails</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Append 101</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>For each subsequence length, keep the smallest possible ending value.</p></div><figcaption><strong>Read it this way:</strong> Each new smaller value replaces the tail for length 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Keep the smallest possible end for each length.

**Simple idea:** `smallest_end[i]` is the smallest ending value found for an increasing
subsequence of length `i + 1`. Replace the first ending value that is not smaller than
the
new value. A smaller ending value gives future values more room.

```python
from bisect import bisect_left

def length_of_lis(nums: list[int]) -> int:
   smallest_end: list[int] = []
   for num in nums:
      index = bisect_left(smallest_end, num)
      if index == len(smallest_end):
         smallest_end.append(num)
      else:
         smallest_end[index] = num
   return len(smallest_end)
```

**Cost:** $O(n\log n)$ time and $O(n)$ space.
