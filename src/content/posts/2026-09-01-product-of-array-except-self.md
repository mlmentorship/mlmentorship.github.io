---
title: "Product of Array Except Self"
description: "For each position, return the product of all other values. Do not use division."
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

> For each position, return the product of all other values. Do not use division.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:product-of-array-except-self-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="product-of-array-except-self-state-title"><p class="visual-kicker">Two passes, one answer</p><p class="visual-title" id="product-of-array-except-self-state-title">Product of Array Except Self: Combine the information on both sides of the current position</p><div class="coding-visual coding-visual--prefix" data-coding-visual data-coding-mode="prefix" data-coding-slug="product-of-array-except-self" role="group" aria-label="Product of Array Except Self: [1, 2, 3, 4] -&gt; answer at 2 is left 1*2 times right 4. The accumulators describe only values outside the current position, so the current value is excluded."><div class="coding-visual-example"><span>Concrete trace</span><strong>[1, 2, 3, 4] -&gt; answer at 2 is left 1*2 times right 4</strong></div><div class="coding-visual-sketch coding-visual-sketch--prefix"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--input">left</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--state">index</span><span class="coding-sketch-arrow">&larr;</span><span class="coding-sketch-pill coding-sketch-pill--focus">right</span></div><p class="coding-sketch-note">two passes meet at one position without including its own value</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Forward</span><strong>left accumulator</strong><small>Carry everything strictly before the current item.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Store</span><strong>left contribution</strong><small>Write the part that belongs in this answer.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Backward</span><strong>right accumulator</strong><small>Walk from the other side without revisiting the array.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Combine</span><strong>left × right</strong><small>Join both outside contributions at the current position.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The accumulators describe only values outside the current position, so the current value is excluded.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> See each answer as a hole in the array. One pass fills the left side of every hole; the reverse pass supplies the right side or the earlier prefix count. For this problem, hold onto the concrete trace: [1, 2, 3, 4] -&gt; answer at 2 is left 1*2 times right 4.</figcaption></figure>

**Pattern:** Prefix and suffix products.

**Simple idea:** The answer at one position is the product on its left times the product on
its right. First save every left product. Then multiply by each right product.

```python
def product_except_self(nums: list[int]) -> list[int]:
   answer = [1] * len(nums)

   prefix = 1
   for index, num in enumerate(nums):
      answer[index] = prefix
      prefix *= num

   suffix = 1
   for index in range(len(nums) - 1, -1, -1):
      answer[index] *= suffix
      suffix *= nums[index]

   return answer
```

**Cost:** $O(n)$ time and $O(1)$ extra space, not counting the answer.
