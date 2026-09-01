---
title: "Binary Search"
description: "Find a target in a sorted array."
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

> Find a target in a sorted array.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:binary-search-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-search-state-title"><p class="visual-kicker">A shrinking answer space</p><p class="visual-title" id="binary-search-state-title">Binary Search: Discard a half only after a yes-or-no test</p><div class="coding-visual coding-visual--binary" data-coding-visual data-coding-mode="binary" data-coding-slug="binary-search" role="group" aria-label="Binary Search: [1,3,5,7,9], target 7 -&gt; mid 5 rules out the lower half. The answer never leaves the current low-to-high interval."><div class="coding-visual-example"><span>Concrete trace</span><strong>[1,3,5,7,9], target 7 -&gt; mid 5 rules out the lower half</strong></div><div class="coding-visual-sketch coding-visual-sketch--binary"><div class="coding-sketch-array"><span class="coding-sketch-pointer">lo</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell coding-sketch-cell--active">mid</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-pointer">hi</span></div><p class="coding-sketch-note">probe the middle, then discard the side the predicate rules out</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Bound</span><strong>lo ... hi</strong><small>Every possible answer is inside this interval.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Probe</span><strong>mid</strong><small>Test the middle value or candidate answer.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Decide</span><strong>predicate</strong><small>The monotone result says which side can survive.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Keep</span><strong>one half</strong><small>Move one boundary and preserve the answer.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The answer never leaves the current low-to-high interval.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the interval as a promise: everything outside it is already impossible. The midpoint is useful only because the predicate is monotone. For this problem, hold onto the concrete trace: [1,3,5,7,9], target 7 -&gt; mid 5 rules out the lower half.</figcaption></figure>

**Pattern:** Binary search on indices.

**Simple idea:** Compare the middle value with the target. Keep only the half that may still
contain the target.

```python
def binary_search(nums: list[int], target: int) -> int:
   left, right = 0, len(nums) - 1

   while left <= right:
      middle = (left + right) // 2
      if nums[middle] == target:
         return middle
      if nums[middle] < target:
         left = middle + 1
      else:
         right = middle - 1

   return -1
```

**Cost:** $O(\log n)$ time and $O(1)$ space.
