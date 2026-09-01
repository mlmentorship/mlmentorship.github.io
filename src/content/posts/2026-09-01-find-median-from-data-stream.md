---
title: "Find Median From Data Stream"
description: "Add numbers one at a time and return the current median."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

> Add numbers one at a time and return the current median.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:find-median-from-data-stream-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="find-median-from-data-stream-state-title"><p class="visual-kicker">A frontier ordered by value</p><p class="visual-title" id="find-median-from-data-stream-state-title">Find Median From Data Stream: Keep the candidates that can still win</p><div class="coding-visual coding-visual--heap" data-coding-visual data-coding-mode="heap" data-coding-slug="find-median-from-data-stream" role="group" aria-label="Find Median From Data Stream: lower heap holds the smaller half, upper heap the larger half, roots meet at median. The heap root is the next candidate whose priority is safe to process."><div class="coding-visual-example"><span>Concrete trace</span><strong>lower heap holds the smaller half, upper heap the larger half, roots meet at median</strong></div><div class="coding-visual-sketch coding-visual-sketch--heap"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">root: next best</span><div class="coding-sketch-branch"><span class="coding-sketch-node">candidate</span><span class="coding-sketch-node">candidate</span></div></div><p class="coding-sketch-note">the root is exposed while the rest stays as a frontier</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Offer</span><strong>candidate set</strong><small>Put a new value into the frontier.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expose</span><strong>root = next best</strong><small>The heap makes the smallest or largest current item visible.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Trim</span><strong>keep k</strong><small>Discard a candidate that cannot enter the answer.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Advance</span><strong>next candidate</strong><small>Replace the used item and continue the stream.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The heap root is the next candidate whose priority is safe to process.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The heap is not a sorted list. It exposes only the next useful item, while preserving enough frontier state to continue without sorting everything. For this problem, hold onto the concrete trace: lower heap holds the smaller half, upper heap the larger half, roots meet at median.</figcaption></figure>

**Pattern:** Two heaps.

**Simple idea:** A max-heap stores the lower half and a min-heap stores the upper half. Keep
the lower half the same size as the upper half or one item larger. The middle value or
values
are then at the heap roots.

Python has only a min-heap, so negative values create the max-heap.

```python
import heapq

class MedianFinder:
   def __init__(self) -> None:
      self.lower: list[int] = []
      self.upper: list[int] = []

   def add_num(self, num: int) -> None:
      heapq.heappush(self.lower, -num)
      heapq.heappush(self.upper, -heapq.heappop(self.lower))
      if len(self.upper) > len(self.lower):
         heapq.heappush(self.lower, -heapq.heappop(self.upper))

   def find_median(self) -> float:
      if len(self.lower) > len(self.upper):
         return float(-self.lower[0])
      return (-self.lower[0] + self.upper[0]) / 2
```

**Cost:** Adding takes $O(\log n)$ time. Finding the median takes $O(1)$ time. Space is
$O(n)$.
