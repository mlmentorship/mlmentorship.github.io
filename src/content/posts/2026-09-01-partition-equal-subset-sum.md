---
title: "Partition Equal Subset Sum"
description: "Check whether the values can be split into two groups with equal sums."
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

> Check whether the values can be split into two groups with equal sums.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:partition-equal-subset-sum-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="partition-equal-subset-sum-state-title"><p class="visual-kicker">A small state graph</p><p class="visual-title" id="partition-equal-subset-sum-state-title">Partition Equal Subset Sum: Keep the complete answer for each smaller state</p><div class="coding-visual coding-visual--dp" data-coding-visual data-coding-mode="dp" data-coding-slug="partition-equal-subset-sum" role="group" aria-label="Partition Equal Subset Sum: [1,5,11,5] -&gt; total 22, so ask whether sum 11 is reachable. Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes."><div class="coding-visual-example"><span>Concrete trace</span><strong>[1,5,11,5] -&gt; total 22, so ask whether sum 11 is reachable</strong></div><div class="coding-visual-sketch coding-visual-sketch--dp"><div class="coding-sketch-dp-grid"><span class="coding-sketch-cell coding-sketch-cell--state">base</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">smaller</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell coding-sketch-cell--active">current</span></div><p class="coding-sketch-note">each cell is a complete answer to one smaller question</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Base</span><strong>known state</strong><small>Initialize the smallest solvable problem.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Read</span><strong>earlier answers</strong><small>Look only at states the transition depends on.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Build</span><strong>current state</strong><small>Choose, count, or combine those answers.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Compress</span><strong>rolling memory</strong><small>Discard old states that no future step needs.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Treat the table as a map of smaller questions. The recurrence is the arrow between states; space optimization is safe only after the dependencies are visible. For this problem, hold onto the concrete trace: [1,5,11,5] -&gt; total 22, so ask whether sum 11 is reachable.</figcaption></figure>

**Pattern:** Subset-sum DP.

**Simple idea:** The wanted sum is half the total. Keep every sum that can be made from the
values processed so far.

```python
def can_partition(nums: list[int]) -> bool:
   total = sum(nums)
   if total % 2:
      return False

   target = total // 2
   possible = {0}
   for num in nums:
      possible |= {value + num for value in possible if value + num <= target}
   return target in possible
```

**Cost:** $O(n \times target)$ time and $O(target)$ space.
