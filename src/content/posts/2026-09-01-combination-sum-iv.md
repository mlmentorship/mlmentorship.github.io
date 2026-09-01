---
title: "Combination Sum IV"
description: "Count ordered sequences of values that add to the target."
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

> Count ordered sequences of values that add to the target.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:combination-sum-iv-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="combination-sum-iv-state-title"><p class="visual-kicker">A small state graph</p><p class="visual-title" id="combination-sum-iv-state-title">Combination Sum IV: Keep the complete answer for each smaller state</p><div class="coding-visual coding-visual--dp" data-coding-visual data-coding-mode="dp" data-coding-slug="combination-sum-iv" role="group" aria-label="Combination Sum IV: target 4 with 1,2 -&gt; count sequences by choosing their final number. Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes."><div class="coding-visual-example"><span>Concrete trace</span><strong>target 4 with 1,2 -&gt; count sequences by choosing their final number</strong></div><div class="coding-visual-sketch coding-visual-sketch--dp"><div class="coding-sketch-dp-grid"><span class="coding-sketch-cell coding-sketch-cell--state">base</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">smaller</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell coding-sketch-cell--active">current</span></div><p class="coding-sketch-note">each cell is a complete answer to one smaller question</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Base</span><strong>known state</strong><small>Initialize the smallest solvable problem.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Read</span><strong>earlier answers</strong><small>Look only at states the transition depends on.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Build</span><strong>current state</strong><small>Choose, count, or combine those answers.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Compress</span><strong>rolling memory</strong><small>Discard old states that no future step needs.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Treat the table as a map of smaller questions. The recurrence is the arrow between states; space optimization is safe only after the dependencies are visible. For this problem, hold onto the concrete trace: target 4 with 1,2 -&gt; count sequences by choosing their final number.</figcaption></figure>

**Pattern:** Counting DP.

**Simple idea:** To build `total`, place each possible value last. Add the number of ordered
ways to build `total - value`. Start with one way to build zero: choose nothing.

```python
def combination_sum_four(nums: list[int], target: int) -> int:
   ways = [1] + [0] * target
   for total in range(1, target + 1):
      for num in nums:
         if num <= total:
            ways[total] += ways[total - num]
   return ways[target]
```

**Cost:** $O(target \times n)$ time and $O(target)$ space.
