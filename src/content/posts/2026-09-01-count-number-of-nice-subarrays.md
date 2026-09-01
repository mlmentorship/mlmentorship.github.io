---
title: "Count Number of Nice Subarrays"
description: "Count subarrays that contain exactly `k` odd numbers."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Foundation"
priority: "Core"
aliases: []
prerequisites: []
---

> Count subarrays that contain exactly `k` odd numbers.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:count-number-of-nice-subarrays-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="count-number-of-nice-subarrays-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="count-number-of-nice-subarrays-state-title">Count Number of Nice Subarrays: Count exact odd counts by subtracting two at-most windows.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="count-number-of-nice-subarrays" role="group" tabindex="0" aria-label="Count Number of Nice Subarrays: Count exact odd counts by subtracting two at-most windows."><div class="coding-visual-example"><span>Input and goal</span><strong>Count subarrays that contain exactly `k` odd numbers.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="At most 3 odd values"><div class="coding-trace-frame-heading"><span>At most 3 odd values</span><strong>The final valid window begins at index 1; the full array has four odd values.</strong></div><div class="coding-trace-dual-window"><div class="coding-trace-window-row"><span class="coding-trace-label">at most 3</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">2</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">1</span><b>14 subarrays</b></div><div class="coding-trace-window-row"><span class="coding-trace-label">at most 2</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell is-inside">2</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">1</span><b>12 subarrays</b></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="At most 2 odd values"><div class="coding-trace-frame-heading"><span>At most 2 odd values</span><strong>The second left boundary moves to index 2, leaving two odd values in the final window.</strong></div><div class="coding-trace-dual-window"><div class="coding-trace-window-row"><span class="coding-trace-label">at most 3</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">2</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">1</span><b>14</b></div><div class="coding-trace-window-row"><span class="coding-trace-label">at most 2</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell is-inside">2</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">1</span><b>12</b></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Subtract the counts"><div class="coding-trace-frame-heading"><span>Subtract the counts</span><strong>Exactly 3 odds = at_most(3) - at_most(2) = 14 - 12 = 2.</strong></div><div class="coding-trace-dual-window"><div class="coding-trace-window-row"><span class="coding-trace-label">at most 3</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">2</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">1</span><b>14</b></div><div class="coding-trace-window-row"><span class="coding-trace-label">at most 2</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell">1</span><span class="coding-trace-window-cell is-inside">2</span><span class="coding-trace-window-cell is-inside">1</span><span class="coding-trace-window-cell is-inside">1</span><b>12</b></div></div><div class="coding-trace-meta"><span><b>result</b>2 nice subarrays</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>At most 3 odd values</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>At most 2 odd values</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Subtract the counts</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Count exact odd counts by subtracting two at-most windows.</p></div><figcaption><strong>Read it this way:</strong> The final valid window begins at index 1; the full array has four odd values. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Exact count from two sliding windows.

**Simple idea:** Counting exactly `k` can be hard. Count subarrays with at most `k`, then
remove subarrays with at most `k - 1`.

`exactly(k) = at_most(k) - at_most(k - 1)`

```python
def number_of_nice_subarrays(nums: list[int], odd_count: int) -> int:
   def at_most(limit: int) -> int:
      if limit < 0:
         return 0

      left = 0
      total = 0
      for right, num in enumerate(nums):
         limit -= num % 2
         while limit < 0:
            limit += nums[left] % 2
            left += 1
         total += right - left + 1
      return total

   return at_most(odd_count) - at_most(odd_count - 1)
```

**Cost:** $O(n)$ time and $O(1)$ space.
