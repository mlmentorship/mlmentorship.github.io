---
title: "Subarray Sum Equals K"
description: "Count continuous subarrays whose sum equals the target."
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

> Count continuous subarrays whose sum equals the target.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:subarray-sum-equals-k-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="subarray-sum-equals-k-state-title"><p class="visual-kicker">Two passes, one answer</p><p class="visual-title" id="subarray-sum-equals-k-state-title">Subarray Sum Equals K: Combine the information on both sides of the current position</p><div class="coding-visual coding-visual--prefix" data-coding-visual data-coding-mode="prefix" data-coding-slug="subarray-sum-equals-k" role="group" aria-label="Subarray Sum Equals K: [1, 2, 1], k=3 -&gt; prefix 3 looks for earlier prefix 0. The accumulators describe only values outside the current position, so the current value is excluded."><div class="coding-visual-example"><span>Concrete trace</span><strong>[1, 2, 1], k=3 -&gt; prefix 3 looks for earlier prefix 0</strong></div><div class="coding-visual-sketch coding-visual-sketch--prefix"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--input">left</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--state">index</span><span class="coding-sketch-arrow">&larr;</span><span class="coding-sketch-pill coding-sketch-pill--focus">right</span></div><p class="coding-sketch-note">two passes meet at one position without including its own value</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Forward</span><strong>left accumulator</strong><small>Carry everything strictly before the current item.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Store</span><strong>left contribution</strong><small>Write the part that belongs in this answer.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Backward</span><strong>right accumulator</strong><small>Walk from the other side without revisiting the array.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Combine</span><strong>left × right</strong><small>Join both outside contributions at the current position.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The accumulators describe only values outside the current position, so the current value is excluded.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> See each answer as a hole in the array. One pass fills the left side of every hole; the reverse pass supplies the right side or the earlier prefix count. For this problem, hold onto the concrete trace: [1, 2, 1], k=3 -&gt; prefix 3 looks for earlier prefix 0.</figcaption></figure>

**Pattern:** Prefix sum plus hash map.

**Simple idea:** If the current prefix sum is `p`, a wanted subarray starts after an earlier
prefix sum of `p - target`. Store how many times each earlier prefix sum appeared.

This is Two Sum on prefix sums.

```python
def subarray_sum(nums: list[int], target: int) -> int:
   prefix_count = {0: 1}
   prefix = 0
   answer = 0

   for num in nums:
      prefix += num
      answer += prefix_count.get(prefix - target, 0)
      prefix_count[prefix] = prefix_count.get(prefix, 0) + 1

   return answer
```

**Cost:** $O(n)$ time and $O(n)$ space.

**Why the map starts with `{0: 1}`:** A prefix that already equals the target forms a valid
subarray starting at index 0.
