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
<figure class="learning-figure coding-visual-figure" aria-labelledby="maximum-product-subarray-state-title"><p class="visual-kicker">Keep both signs alive</p><p class="visual-title" id="maximum-product-subarray-state-title">Maximum Product Subarray: A negative value can swap the best and worst futures</p><div class="coding-visual coding-visual--extrema" data-coding-visual data-coding-mode="extrema" data-coding-slug="maximum-product-subarray" role="group" aria-label="Maximum Product Subarray: [-2,3,-4] -&gt; the negative minimum becomes the positive maximum. Both the maximum and minimum product ending here are available for the next value."><div class="coding-visual-example"><span>Concrete trace</span><strong>[-2,3,-4] -&gt; the negative minimum becomes the positive maximum</strong></div><div class="coding-visual-sketch coding-visual-sketch--extrema"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">minimum</span><span class="coding-sketch-arrow">&harr;</span><span class="coding-sketch-pill coding-sketch-pill--focus">negative?</span><span class="coding-sketch-arrow">&harr;</span><span class="coding-sketch-pill coding-sketch-pill--active">maximum</span></div><p class="coding-sketch-note">a negative input can swap which extreme wins next</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Hold</span><strong>max and min</strong><small>Keep both extremes ending at the current position.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Flip</span><strong>negative multiplier</strong><small>A negative value exchanges their future roles.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Extend</span><strong>or restart</strong><small>Multiply the old extremes or begin at this value.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>largest product</strong><small>Save the best ending value seen so far.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Both the maximum and minimum product ending here are available for the next value.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The minimum is not discarded as bad news. One more negative value can turn it into the maximum, so the visual keeps both futures alive. For this problem, hold onto the concrete trace: [-2,3,-4] -&gt; the negative minimum becomes the positive maximum.</figcaption></figure>

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
