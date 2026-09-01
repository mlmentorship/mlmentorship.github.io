---
title: "Maximum Product Subarray"
description: "Find the largest product of a nonempty continuous subarray."
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

> Find the largest product of a nonempty continuous subarray.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:maximum-product-subarray-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="maximum-product-subarray-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="maximum-product-subarray-state-title">Maximum Product Subarray: Keep both product extremes because a negative number can swap their roles.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="maximum-product-subarray" role="group" aria-label="Maximum Product Subarray: Keep both product extremes because a negative number can swap their roles."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the largest product of a nonempty continuous subarray.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Start both extremes"><div class="coding-trace-frame-heading"><span>Start both extremes</span><strong>At 2 and 3, max and min ending products are both positive.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">max=6,min=3</span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span></span></div><div class="coding-trace-meta"><span><b>max</b>6</span><span><b>min</b>3</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="A negative flips them"><div class="coding-trace-frame-heading"><span>A negative flips them</span><strong>At -2, restarting at -2 is the maximum while the carried products become negative.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">flip</span><span class="coding-trace-array-cell">-2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span></span></div><div class="coding-trace-meta"><span><b>max</b>-2</span><span><b>min</b>-12</span><span><b>detail</b>candidates: -2, 6*-2, 3*-2</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Recover with another negative"><div class="coding-trace-frame-heading"><span>Recover with another negative</span><strong>The best product 6 comes from [2,3], while later values are checked the same way.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">best</span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">best</span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span></span></div><div class="coding-trace-meta"><span><b>result</b>6</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start both extremes</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>A negative flips them</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Recover with another negative</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Keep both product extremes because a negative number can swap their roles.</p></div><figcaption><strong>Read it this way:</strong> At 2 and 3, max and min ending products are both positive. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** DP with current maximum and minimum.

**Simple idea:** A negative value can turn the smallest negative product into the largest
positive product. Keep both extremes. Swap them before multiplying by a negative value.

```python
def max_product_subarray(nums: list[int]) -> int:
   current_max = current_min = best = nums[0]

   for num in nums[1:]:
      if num < 0:
         current_max, current_min = current_min, current_max
      current_max = max(num, current_max * num)
      current_min = min(num, current_min * num)
      best = max(best, current_max)
   return best
```

**Cost:** $O(n)$ time and $O(1)$ space.
