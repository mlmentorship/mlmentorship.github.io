---
title: "House Robber II"
description: "Houses form a circle, so the first and last houses are neighbors."
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

> Houses form a circle, so the first and last houses are neighbors.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:house-robber-ii-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="house-robber-ii-state-title"><p class="visual-kicker">A small state graph</p><p class="visual-title" id="house-robber-ii-state-title">House Robber II: Keep the complete answer for each smaller state</p><div class="coding-visual coding-visual--dp" data-coding-visual data-coding-mode="dp" data-coding-slug="house-robber-ii" role="group" aria-label="House Robber II: circle [2,3,2] -&gt; solve without first and without last, then take max. Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes."><div class="coding-visual-example"><span>Concrete trace</span><strong>circle [2,3,2] -&gt; solve without first and without last, then take max</strong></div><div class="coding-visual-sketch coding-visual-sketch--dp"><div class="coding-sketch-dp-grid"><span class="coding-sketch-cell coding-sketch-cell--state">base</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">smaller</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell coding-sketch-cell--active">current</span></div><p class="coding-sketch-note">each cell is a complete answer to one smaller question</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Base</span><strong>known state</strong><small>Initialize the smallest solvable problem.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Read</span><strong>earlier answers</strong><small>Look only at states the transition depends on.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Build</span><strong>current state</strong><small>Choose, count, or combine those answers.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Compress</span><strong>rolling memory</strong><small>Discard old states that no future step needs.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Treat the table as a map of smaller questions. The recurrence is the arrow between states; space optimization is safe only after the dependencies are visible. For this problem, hold onto the concrete trace: circle [2,3,2] -&gt; solve without first and without last, then take max.</figcaption></figure>

**Pattern:** Reduce a circle to two linear DP problems.

**Simple idea:** A valid answer cannot take both the first and last houses. Solve once
without the last house and once without the first house. Keep the larger answer.

```python
def house_robber_two(nums: list[int]) -> int:
   def rob(line: list[int]) -> int:
      two_back = one_back = 0
      for money in line:
         two_back, one_back = one_back, max(one_back, two_back + money)
      return one_back

   if len(nums) == 1:
      return nums[0]
   return max(rob(nums[:-1]), rob(nums[1:]))
```

**Cost:** $O(n)$ time and $O(n)$ space from the two slices. Index ranges can reduce extra
space to $O(1)$.
