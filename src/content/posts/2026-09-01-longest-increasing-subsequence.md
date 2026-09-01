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
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-increasing-subsequence-state-title"><p class="visual-kicker">A small state graph</p><p class="visual-title" id="longest-increasing-subsequence-state-title">Longest Increasing Subsequence: Keep the complete answer for each smaller state</p><div class="coding-visual coding-visual--dp" data-coding-visual data-coding-mode="dp" data-coding-slug="longest-increasing-subsequence" role="group" aria-label="Longest Increasing Subsequence: 10,9,2,5,3,7,101 -&gt; tails become 2,3,7,101. Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes."><div class="coding-visual-example"><span>Concrete trace</span><strong>10,9,2,5,3,7,101 -&gt; tails become 2,3,7,101</strong></div><div class="coding-visual-sketch coding-visual-sketch--dp"><div class="coding-sketch-dp-grid"><span class="coding-sketch-cell coding-sketch-cell--state">base</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">smaller</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell coding-sketch-cell--active">current</span></div><p class="coding-sketch-note">each cell is a complete answer to one smaller question</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Base</span><strong>known state</strong><small>Initialize the smallest solvable problem.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Read</span><strong>earlier answers</strong><small>Look only at states the transition depends on.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Build</span><strong>current state</strong><small>Choose, count, or combine those answers.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Compress</span><strong>rolling memory</strong><small>Discard old states that no future step needs.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Treat the table as a map of smaller questions. The recurrence is the arrow between states; space optimization is safe only after the dependencies are visible. For this problem, hold onto the concrete trace: 10,9,2,5,3,7,101 -&gt; tails become 2,3,7,101.</figcaption></figure>

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
