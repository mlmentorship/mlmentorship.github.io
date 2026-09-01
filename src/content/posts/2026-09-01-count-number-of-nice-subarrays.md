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
<figure class="learning-figure coding-visual-figure" aria-labelledby="count-number-of-nice-subarrays-state-title"><p class="visual-kicker">A moving range</p><p class="visual-title" id="count-number-of-nice-subarrays-state-title">Count Number of Nice Subarrays: Grow until valid, then shrink until necessary</p><div class="coding-visual coding-visual--window" data-coding-visual data-coding-mode="window" data-coding-slug="count-number-of-nice-subarrays" role="group" aria-label="Count Number of Nice Subarrays: [1,1,2,1,1], k=3 -&gt; prefix odd counts 0,1,2,2,3,4. The current window has exactly the state needed to decide whether it is valid."><div class="coding-visual-example"><span>Concrete trace</span><strong>[1,1,2,1,1], k=3 -&gt; prefix odd counts 0,1,2,2,3,4</strong></div><div class="coding-visual-sketch coding-visual-sketch--window"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-cell coding-sketch-cell--active">active</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">the active bracket grows for evidence and shrinks when its state is sufficient</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Extend</span><strong>L ... R</strong><small>Move the right edge to include new evidence.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Measure</span><strong>window state</strong><small>Update counts, sum, or the required matches.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Tighten</span><strong>advance L</strong><small>Remove the oldest item while validity survives.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best valid range</strong><small>Save the shortest, longest, or counted window.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The current window has exactly the state needed to decide whether it is valid.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The two edges are not guesses. The right edge gathers enough evidence; the left edge removes anything no longer needed, so each item enters and leaves once. For this problem, hold onto the concrete trace: [1,1,2,1,1], k=3 -&gt; prefix odd counts 0,1,2,2,3,4.</figcaption></figure>

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
