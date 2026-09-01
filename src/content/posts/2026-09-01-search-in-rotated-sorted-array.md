---
title: "Search in Rotated Sorted Array"
description: "Find a target in a sorted array that was rotated once."
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

> Find a target in a sorted array that was rotated once.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:search-in-rotated-sorted-array-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="search-in-rotated-sorted-array-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="search-in-rotated-sorted-array-state-title">Search in Rotated Sorted Array: Use the sorted half to decide which side of the rotation can contain the target.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="search-in-rotated-sorted-array" role="group" aria-label="Search in Rotated Sorted Array: Use the sorted half to decide which side of the rotation can contain the target."><div class="coding-visual-example"><span>Input and goal</span><strong>Find a target in a sorted array that was rotated once.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Identify a sorted half"><div class="coding-trace-frame-heading"><span>Identify a sorted half</span><strong>For [4,5,6,7,0,1,2], the left half 4..7 is sorted.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">L</span><span class="coding-trace-array-cell">4</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">mid</span><span class="coding-trace-array-cell">7</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">R</span><span class="coding-trace-array-cell">2</span></span></div><div class="coding-trace-meta"><span><b>detail</b>left half sorted</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Choose the other half"><div class="coding-trace-frame-heading"><span>Choose the other half</span><strong>Target 0 is not inside 4..7, so discard the sorted left half.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">L</span><span class="coding-trace-array-cell">0</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">mid</span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">R</span><span class="coding-trace-array-cell">2</span></span></div><div class="coding-trace-meta"><span><b>detail</b>search right half</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Find the target"><div class="coding-trace-frame-heading"><span>Find the target</span><strong>The right half reaches 0 at index 4.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">found</span><span class="coding-trace-array-cell">0</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span></div><div class="coding-trace-meta"><span><b>result</b>index 4</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Identify a sorted half</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Choose the other half</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Find the target</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Use the sorted half to decide which side of the rotation can contain the target.</p></div><figcaption><strong>Read it this way:</strong> For [4,5,6,7,0,1,2], the left half 4..7 is sorted. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Binary search with one sorted half.

**Simple idea:** At least one half around the middle is sorted. Find that half. If the target
fits inside its values, search it. Otherwise search the other half.

```python
def search_rotated(nums: list[int], target: int) -> int:
   left, right = 0, len(nums) - 1

   while left <= right:
      middle = (left + right) // 2
      if nums[middle] == target:
         return middle

      if nums[left] <= nums[middle]:
         if nums[left] <= target < nums[middle]:
            right = middle - 1
         else:
            left = middle + 1
      elif nums[middle] < target <= nums[right]:
         left = middle + 1
      else:
         right = middle - 1

   return -1
```

**Cost:** $O(\log n)$ time and $O(1)$ space.
