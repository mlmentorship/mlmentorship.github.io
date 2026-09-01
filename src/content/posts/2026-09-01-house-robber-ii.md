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
<figure class="learning-figure coding-visual-figure" aria-labelledby="house-robber-ii-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="house-robber-ii-state-title">House Robber II: A circular solution is the larger of two lines: exclude the first house or exclude the last.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="house-robber-ii" role="group" aria-label="House Robber II: A circular solution is the larger of two lines: exclude the first house or exclude the last."><div class="coding-visual-example"><span>Input and goal</span><strong>Houses form a circle, so the first and last houses are neighbors.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Break the circle"><div class="coding-trace-frame-heading"><span>Break the circle</span><strong>Taking both first and last is forbidden, so solve two linear ranges.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">exclude in case B</span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">exclude in case A</span><span class="coding-trace-array-cell">2</span></span></div><div class="coding-trace-meta"><span><b>cases</b>houses[0:-1] and houses[1:]</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Solve each line"><div class="coding-trace-frame-heading"><span>Solve each line</span><strong>Case A [2,3] gives 3. Case B [3,2] gives 3.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">case</th><th scope="col">houses</th><th scope="col">best</th></tr></thead><tbody><tr><td class="is-active">A</td><td class="is-active">[2,3]</td><td class="">3</td></tr><tr><td class="">B</td><td class="">[3,2]</td><td class="">3</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Choose the larger result"><div class="coding-trace-frame-heading"><span>Choose the larger result</span><strong>Both cases tie at 3, which is the circular answer.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">take</span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span></div><div class="coding-trace-meta"><span><b>result</b>3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Break the circle</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Solve each line</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Choose the larger result</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A circular solution is the larger of two lines: exclude the first house or exclude the last.</p></div><figcaption><strong>Read it this way:</strong> Taking both first and last is forbidden, so solve two linear ranges. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
