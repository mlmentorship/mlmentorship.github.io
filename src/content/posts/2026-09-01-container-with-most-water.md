---
title: "Container With Most Water"
description: "Pick two heights that hold the most water."
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

> Pick two heights that hold the most water.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:container-with-most-water-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="container-with-most-water-state-title"><p class="visual-kicker">Two ends, one proof</p><p class="visual-title" id="container-with-most-water-state-title">Container With Most Water: Move an endpoint only when the other choice cannot help</p><div class="coding-visual coding-visual--two-pointer" data-coding-visual data-coding-mode="two-pointer" data-coding-slug="container-with-most-water" role="group" aria-label="Container With Most Water: [1, 8, 6, 2, 5, 4, 8, 3, 7] -&gt; move the shorter wall inward. Everything outside the two pointers has been checked or proven unable to improve the answer."><div class="coding-visual-example"><span>Concrete trace</span><strong>[1, 8, 6, 2, 5, 4, 8, 3, 7] -&gt; move the shorter wall inward</strong></div><div class="coding-visual-sketch coding-visual-sketch--two-pointer"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell">candidate</span><span class="coding-sketch-cell coding-sketch-cell--active">pair</span><span class="coding-sketch-cell">candidate</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">compare both ends, then move the limiting side</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Arrange</span><strong>sorted or bounded</strong><small>Put the candidates in an order that supports comparison.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Compare</span><strong>left + right</strong><small>Measure the pair or the container formed by both ends.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Move</span><strong>provably weaker end</strong><small>Discard the side that cannot improve the answer.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>valid pair or best area</strong><small>Save the result before narrowing the search.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Everything outside the two pointers has been checked or proven unable to improve the answer.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The pointers are a proof boundary. The next move is safe only because one endpoint is the limiting factor or the sorted sum has the wrong sign. For this problem, hold onto the concrete trace: [1, 8, 6, 2, 5, 4, 8, 3, 7] -&gt; move the shorter wall inward.</figcaption></figure>

**Pattern:** Two pointers at opposite ends.

**Simple idea:** Width gets smaller after every move. Move the shorter wall because the
shorter wall limits the current area. Moving the taller wall cannot improve that limit.

```python
def max_area(heights: list[int]) -> int:
   left, right = 0, len(heights) - 1
   best = 0

   while left < right:
      height = min(heights[left], heights[right])
      best = max(best, height * (right - left))
      if heights[left] < heights[right]:
         left += 1
      else:
         right -= 1

   return best
```

**Cost:** $O(n)$ time and $O(1)$ space.
