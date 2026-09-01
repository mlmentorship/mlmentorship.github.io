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
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-search-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="binary-search-state-title">Binary Search: Keep the sorted half that can still contain the target.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="binary-search" role="group" aria-label="Binary Search: Keep the sorted half that can still contain the target."><div class="coding-visual-example"><span>Input and goal</span><strong>Find a target in a sorted array.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Probe the middle"><div class="coding-trace-frame-heading"><span>Probe the middle</span><strong>The middle value 5 is below target 7, so the lower half is finished.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">discard</span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">discard</span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">mid</span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">9</span></span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Narrow the interval"><div class="coding-trace-frame-heading"><span>Narrow the interval</span><strong>The remaining interval is [7, 9].</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">lo</span><span class="coding-trace-array-cell">7</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">hi</span><span class="coding-trace-array-cell">9</span></span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Hit the target"><div class="coding-trace-frame-heading"><span>Hit the target</span><strong>The next middle is 7, at index 3.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">found</span><span class="coding-trace-array-cell">7</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">9</span></span></div><div class="coding-trace-meta"><span><b>result</b>index 3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Probe the middle</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Narrow the interval</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Hit the target</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Keep the sorted half that can still contain the target.</p></div><figcaption><strong>Read it this way:</strong> The middle value 5 is below target 7, so the lower half is finished. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
