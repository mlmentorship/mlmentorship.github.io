---
title: "Maximum Subarray"
description: "Find the largest sum of a nonempty continuous subarray."
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

> Find the largest sum of a nonempty continuous subarray.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:maximum-subarray-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="maximum-subarray-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="maximum-subarray-state-title">Maximum Subarray: Discard a negative running prefix before extending a future subarray.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="maximum-subarray" role="group" aria-label="Maximum Subarray: Discard a negative running prefix before extending a future subarray."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the largest sum of a nonempty continuous subarray.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Carry a running sum"><div class="coding-trace-frame-heading"><span>Carry a running sum</span><strong>At 1, the negative prefix -2 is worse than starting a new subarray.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-warning" role="listitem"><span class="coding-trace-array-mark">drop</span><span class="coding-trace-array-cell">-2</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">start</span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span></div><div class="coding-trace-meta"><span><b>current</b>1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Extend the best ending here"><div class="coding-trace-frame-heading"><span>Extend the best ending here</span><strong>Starting at 4, the running sum grows through -1, 2, and 1.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">start</span><span class="coding-trace-array-cell">4</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">best ending</span><span class="coding-trace-array-cell">1</span></span></div><div class="coding-trace-meta"><span><b>current</b>6</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Keep the global best"><div class="coding-trace-frame-heading"><span>Keep the global best</span><strong>The maximum subarray is [4,-1,2,1] with sum 6.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-3</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">best</span><span class="coding-trace-array-cell">4</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">best</span><span class="coding-trace-array-cell">-1</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">best</span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">best</span><span class="coding-trace-array-cell">1</span></span></div><div class="coding-trace-meta"><span><b>result</b>6</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Carry a running sum</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Extend the best ending here</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Keep the global best</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Discard a negative running prefix before extending a future subarray.</p></div><figcaption><strong>Read it this way:</strong> At 1, the negative prefix -2 is worse than starting a new subarray. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** One-dimensional DP, also called Kadane's algorithm.

**Simple idea:** At each value, choose whether to start a new subarray or extend the current
one. A negative earlier total is not worth carrying forward.

```python
def max_subarray(nums: list[int]) -> int:
   current = best = nums[0]
   for num in nums[1:]:
      current = max(num, current + num)
      best = max(best, current)
   return best
```

**Cost:** $O(n)$ time and $O(1)$ space.
