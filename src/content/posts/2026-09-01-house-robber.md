---
title: "House Robber"
description: "Find the most money that can be taken without choosing neighboring houses."
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

> Find the most money that can be taken without choosing neighboring houses.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:house-robber-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="house-robber-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="house-robber-state-title">House Robber: At each house, choose between skipping it and taking it after the previous house.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="house-robber" role="group" tabindex="0" aria-label="House Robber: At each house, choose between skipping it and taking it after the previous house."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the most money that can be taken without choosing neighboring houses.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Before any house"><div class="coding-trace-frame-heading"><span>Before any house</span><strong>The best totals two houses back and one house back are both zero.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-current">current</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">9</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>states</b>two_back=0, one_back=0</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Compare at 9"><div class="coding-trace-frame-heading"><span>Compare at 9</span><strong>Skip 9 gives 7; take 9 gives 0 + 9. Keep 11 after the first three houses.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-take">take</span><span class="coding-trace-array-cell">9</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>states</b>skip=7, take=11, best=11</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Finish the line"><div class="coding-trace-frame-heading"><span>Finish the line</span><strong>The best non-adjacent selection is 2 + 9 + 1 = 12.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-take">take</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-take">take</span><span class="coding-trace-array-cell">9</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="marker-take">take</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>result</b>12</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Before any house</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Compare at 9</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Finish the line</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>At each house, choose between skipping it and taking it after the previous house.</p></div><figcaption><strong>Read it this way:</strong> The best totals two houses back and one house back are both zero. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** DP with two saved values.

**State:** Best answer one house back and two houses back.

**Simple idea:** At each house, choose the better result: skip it, or take it plus the best
answer from two houses back.

```python
def house_robber(nums: list[int]) -> int:
   two_houses_back = 0
   one_house_back = 0

   for money in nums:
      current = max(one_house_back, two_houses_back + money)
      two_houses_back, one_house_back = one_house_back, current

   return one_house_back
```

**Cost:** $O(n)$ time and $O(1)$ space.
