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
<figure class="learning-figure coding-visual-figure" aria-labelledby="maximum-subarray-state-title"><p class="visual-kicker">One pass, running best</p><p class="visual-title" id="maximum-subarray-state-title">Maximum Subarray: Carry the smallest, largest, or best state seen so far</p><div class="coding-visual coding-visual--running" data-coding-visual data-coding-mode="running" data-coding-slug="maximum-subarray" role="group" aria-label="Maximum Subarray: [-2,1,-3,4,-1,2,1] -&gt; discard the negative prefix before 4. The carried state is the complete summary needed to make the next position optimal."><div class="coding-visual-example"><span>Concrete trace</span><strong>[-2,1,-3,4,-1,2,1] -&gt; discard the negative prefix before 4</strong></div><div class="coding-visual-sketch coding-visual-sketch--running"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">best so far</span><span class="coding-sketch-arrow">&larr;</span><span class="coding-sketch-pill coding-sketch-pill--input">current</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--focus">new best?</span></div><p class="coding-sketch-note">the carried summary is enough to judge the next value</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Observe</span><strong>current value</strong><small>Read the next price, sum, or candidate.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Carry</span><strong>state so far</strong><small>Keep the summary future positions can use.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Update</span><strong>best decision</strong><small>Compare starting fresh, extending, buying, or selling.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best answer</strong><small>Preserve the strongest result seen anywhere.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The carried state is the complete summary needed to make the next position optimal.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The scan does not remember every earlier value. It remembers the one summary that gives every future position its best possible continuation. For this problem, hold onto the concrete trace: [-2,1,-3,4,-1,2,1] -&gt; discard the negative prefix before 4.</figcaption></figure>

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
