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
<figure class="learning-figure coding-visual-figure" aria-labelledby="partition-equal-subset-sum-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="partition-equal-subset-sum-state-title">Partition Equal Subset Sum: Reach half the total; every reachable sum is a state.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="partition-equal-subset-sum" role="group" aria-label="Partition Equal Subset Sum: Reach half the total; every reachable sum is a state."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether the values can be split into two groups with equal sums.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Set the target"><div class="coding-trace-frame-heading"><span>Set the target</span><strong>Total is 22, so the wanted subset sum is 11.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">target 11</span><span class="coding-trace-array-cell">11</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">16</span></span></div><div class="coding-trace-meta"><span><b>target</b>11</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Add reachable sums"><div class="coding-trace-frame-heading"><span>Add reachable sums</span><strong>After processing 1, 5, and 11, the set contains 11.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">reachable</span><span class="coding-trace-array-cell">11</span></span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Accept the partition"><div class="coding-trace-frame-heading"><span>Accept the partition</span><strong>A subset totals 11, so the remaining values also total 11.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">sum 11</span><span class="coding-trace-array-cell">[1,5,5]</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">sum 11</span><span class="coding-trace-array-cell">[11]</span></span></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Set the target</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Add reachable sums</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Accept the partition</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Reach half the total; every reachable sum is a state.</p></div><figcaption><strong>Read it this way:</strong> Total is 22, so the wanted subset sum is 11. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
