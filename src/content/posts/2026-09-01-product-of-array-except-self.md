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
<figure class="learning-figure coding-visual-figure" aria-labelledby="product-of-array-except-self-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="product-of-array-except-self-state-title">Product of Array Except Self: Build each answer from the product to its left and the product to its right.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="product-of-array-except-self" role="group" aria-label="Product of Array Except Self: Build each answer from the product to its left and the product to its right."><div class="coding-visual-example"><span>Input and goal</span><strong>For each position, return the product of all other values. Do not use division.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Write left products"><div class="coding-trace-frame-heading"><span>Write left products</span><strong>At each index, store the product strictly before it.</strong></div><div class="coding-trace-prefix"><div class="coding-trace-prefix-row"><span class="coding-trace-label">input</span><span class="coding-trace-prefix-cell is-active">1</span><span class="coding-trace-prefix-cell">2</span><span class="coding-trace-prefix-cell">3</span><span class="coding-trace-prefix-cell">4</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">left</span><span class="coding-trace-prefix-cell is-active">1</span><span class="coding-trace-prefix-cell">1</span><span class="coding-trace-prefix-cell">2</span><span class="coding-trace-prefix-cell">6</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">right</span><span class="coding-trace-prefix-cell is-active">-</span><span class="coding-trace-prefix-cell">-</span><span class="coding-trace-prefix-cell">-</span><span class="coding-trace-prefix-cell">-</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">answer</span><span class="coding-trace-prefix-cell is-active">1</span><span class="coding-trace-prefix-cell">1</span><span class="coding-trace-prefix-cell">2</span><span class="coding-trace-prefix-cell">6</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Walk back from the right"><div class="coding-trace-frame-heading"><span>Walk back from the right</span><strong>A suffix product is multiplied into each saved prefix product.</strong></div><div class="coding-trace-prefix"><div class="coding-trace-prefix-row"><span class="coding-trace-label">input</span><span class="coding-trace-prefix-cell">1</span><span class="coding-trace-prefix-cell">2</span><span class="coding-trace-prefix-cell is-active">3</span><span class="coding-trace-prefix-cell">4</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">left</span><span class="coding-trace-prefix-cell">1</span><span class="coding-trace-prefix-cell">1</span><span class="coding-trace-prefix-cell is-active">2</span><span class="coding-trace-prefix-cell">6</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">right</span><span class="coding-trace-prefix-cell">24</span><span class="coding-trace-prefix-cell">12</span><span class="coding-trace-prefix-cell is-active">4</span><span class="coding-trace-prefix-cell">1</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">answer</span><span class="coding-trace-prefix-cell">24</span><span class="coding-trace-prefix-cell">12</span><span class="coding-trace-prefix-cell is-active">8</span><span class="coding-trace-prefix-cell">6</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Exclude the current value"><div class="coding-trace-frame-heading"><span>Exclude the current value</span><strong>Each answer combines everything on both sides and never divides.</strong></div><div class="coding-trace-prefix"><div class="coding-trace-prefix-row"><span class="coding-trace-label">input</span><span class="coding-trace-prefix-cell">1</span><span class="coding-trace-prefix-cell">2</span><span class="coding-trace-prefix-cell">3</span><span class="coding-trace-prefix-cell is-active">4</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">left</span><span class="coding-trace-prefix-cell">1</span><span class="coding-trace-prefix-cell">1</span><span class="coding-trace-prefix-cell">2</span><span class="coding-trace-prefix-cell is-active">6</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">right</span><span class="coding-trace-prefix-cell">24</span><span class="coding-trace-prefix-cell">12</span><span class="coding-trace-prefix-cell">4</span><span class="coding-trace-prefix-cell is-active">1</span></div><div class="coding-trace-prefix-row"><span class="coding-trace-label">answer</span><span class="coding-trace-prefix-cell">24</span><span class="coding-trace-prefix-cell">12</span><span class="coding-trace-prefix-cell">8</span><span class="coding-trace-prefix-cell is-active">6</span></div></div><div class="coding-trace-meta"><span><b>status</b>complete</span><span><b>result</b>[24,12,8,6]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Write left products</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Walk back from the right</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Exclude the current value</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Build each answer from the product to its left and the product to its right.</p></div><figcaption><strong>Read it this way:</strong> At each index, store the product strictly before it. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
