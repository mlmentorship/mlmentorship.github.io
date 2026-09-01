---
title: "Kth Largest Element"
description: "Find the `k`th largest value in an unsorted array."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Core"
aliases: []
prerequisites: []
---

> Find the `k`th largest value in an unsorted array.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:kth-largest-element-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="kth-largest-element-state-title"><p class="visual-kicker">A frontier ordered by value</p><p class="visual-title" id="kth-largest-element-state-title">Kth Largest Element: Keep the candidates that can still win</p><div class="coding-visual coding-visual--heap" data-coding-visual data-coding-mode="heap" data-coding-slug="kth-largest-element" role="group" aria-label="Kth Largest Element: [3,2,1,5,6,4], k=2 -&gt; a size-2 min-heap keeps 5 and 6. The heap root is the next candidate whose priority is safe to process."><div class="coding-visual-example"><span>Concrete trace</span><strong>[3,2,1,5,6,4], k=2 -&gt; a size-2 min-heap keeps 5 and 6</strong></div><div class="coding-visual-sketch coding-visual-sketch--heap"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">root: next best</span><div class="coding-sketch-branch"><span class="coding-sketch-node">candidate</span><span class="coding-sketch-node">candidate</span></div></div><p class="coding-sketch-note">the root is exposed while the rest stays as a frontier</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Offer</span><strong>candidate set</strong><small>Put a new value into the frontier.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expose</span><strong>root = next best</strong><small>The heap makes the smallest or largest current item visible.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Trim</span><strong>keep k</strong><small>Discard a candidate that cannot enter the answer.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Advance</span><strong>next candidate</strong><small>Replace the used item and continue the stream.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The heap root is the next candidate whose priority is safe to process.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The heap is not a sorted list. It exposes only the next useful item, while preserving enough frontier state to continue without sorting everything. For this problem, hold onto the concrete trace: [3,2,1,5,6,4], k=2 -&gt; a size-2 min-heap keeps 5 and 6.</figcaption></figure>

**Pattern:** Min-heap of size `k`.

**Simple idea:** Keep only the largest `k` values seen. The smallest value in that group is
the `k`th largest overall.

```python
import heapq

def find_kth_largest(nums: list[int], k: int) -> int:
   if not 1 <= k <= len(nums):
      raise ValueError("k must name an item in nums")

   heap: list[int] = []
   for num in nums:
      heapq.heappush(heap, num)
      if len(heap) > k:
         heapq.heappop(heap)
   return heap[0]
```

**Cost:** $O(n\log k)$ time and $O(k)$ space.
