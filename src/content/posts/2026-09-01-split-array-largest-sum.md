---
title: "Split Array Largest Sum"
description: "Split an array into `k` nonempty continuous parts. Make the largest part sum as small as possible."
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

> Split an array into `k` nonempty continuous parts. Make the largest part sum as small as possible.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:split-array-largest-sum-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="split-array-largest-sum-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="split-array-largest-sum-state-title">Split Array Largest Sum: Guess a maximum part sum, greedily count required parts, and binary-search the smallest feasible guess.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="split-array-largest-sum" role="group" aria-label="Split Array Largest Sum: Guess a maximum part sum, greedily count required parts, and binary-search the smallest feasible guess."><div class="coding-visual-example"><span>Input and goal</span><strong>Split an array into `k` nonempty continuous parts. Make the largest part sum as small as possible.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Set answer bounds"><div class="coding-trace-frame-heading"><span>Set answer bounds</span><strong>The largest part must be at least max(nums)=10 and at most total 32.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">lo</span><span class="coding-trace-array-cell">10</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">11</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">12</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">...</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">31</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">hi</span><span class="coding-trace-array-cell">32</span></span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Test a limit"><div class="coding-trace-frame-heading"><span>Test a limit</span><strong>With limit 18, greedy cuts [7,2,5] and [10,8], using two parts.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">part 1</span><span class="coding-trace-array-cell">7+2+5=14</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">part 2</span><span class="coding-trace-array-cell">10+8=18</span></span></div><div class="coding-trace-meta"><span><b>parts</b>2 &lt;= k</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return the smallest feasible limit"><div class="coding-trace-frame-heading"><span>Return the smallest feasible limit</span><strong>18 works, while 17 would require three parts.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">17</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">answer</span><span class="coding-trace-array-cell">18</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">19</span></span></div><div class="coding-trace-meta"><span><b>result</b>18</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Set answer bounds</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Test a limit</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the smallest feasible limit</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Guess a maximum part sum, greedily count required parts, and binary-search the smallest feasible guess.</p></div><figcaption><strong>Read it this way:</strong> The largest part must be at least max(nums)=10 and at most total 32. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Binary search on the answer.

**Simple idea:** Guess the largest allowed sum. Start a new part when adding the next value
would pass the guess. If this needs at most `k` parts, the guess works.

```python
def split_array_largest_sum(nums: list[int], parts: int) -> int:
   def parts_needed(limit: int) -> int:
      used = 1
      total = 0
      for num in nums:
         if total + num > limit:
            used += 1
            total = 0
         total += num
      return used

   left, right = max(nums), sum(nums)
   while left < right:
      middle = (left + right) // 2
      if parts_needed(middle) <= parts:
         right = middle
      else:
         left = middle + 1
   return left
```

**Cost:** $O(n\log s)$ time and $O(1)$ space, where $s$ is the sum range searched.
