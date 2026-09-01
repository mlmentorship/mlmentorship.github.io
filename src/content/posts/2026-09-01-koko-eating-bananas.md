---
title: "Koko Eating Bananas"
description: "Find the slowest eating speed that finishes all piles within the time limit."
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

> Find the slowest eating speed that finishes all piles within the time limit.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:koko-eating-bananas-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="koko-eating-bananas-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="koko-eating-bananas-state-title">Koko Eating Bananas: Binary-search the smallest eating speed that finishes within the hour limit.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="koko-eating-bananas" role="group" aria-label="Koko Eating Bananas: Binary-search the smallest eating speed that finishes within the hour limit."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the slowest eating speed that finishes all piles within the time limit.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Test a speed"><div class="coding-trace-frame-heading"><span>Test a speed</span><strong>At speed 4, piles [3,6,7,11] take 1+2+2+3 = 8 hours.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">speed 1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">speed 2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">speed 3</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">test</span><span class="coding-trace-array-cell">speed 4</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">speed 5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">...</span></span></div><div class="coding-trace-meta"><span><b>detail</b>hours = 8; feasible</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Discard an infeasible speed"><div class="coding-trace-frame-heading"><span>Discard an infeasible speed</span><strong>Speed 3 takes 10 hours, so the smallest feasible speed is at least 4. Keep [4,5].</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item trace-tone-warning" role="listitem"><span class="coding-trace-array-mark">test</span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">lo</span><span class="coding-trace-array-cell">4</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">hi</span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">...</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">11</span></span></div><div class="coding-trace-meta"><span><b>detail</b>10 hours &gt; 8</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return the first feasible speed"><div class="coding-trace-frame-heading"><span>Return the first feasible speed</span><strong>Speed 4 is the smallest speed whose hours fit.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">answer</span><span class="coding-trace-array-cell">4</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">...</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">11</span></span></div><div class="coding-trace-meta"><span><b>result</b>4 bananas/hour</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Test a speed</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Discard an infeasible speed</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the first feasible speed</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Binary-search the smallest eating speed that finishes within the hour limit.</p></div><figcaption><strong>Read it this way:</strong> At speed 4, piles [3,6,7,11] take 1+2+2+3 = 8 hours. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Binary search on the answer.

**Simple idea:** Try a speed. If it is fast enough, a faster speed is also fast enough. This
creates one false-to-true boundary. Search for the first true speed.

```python
def min_eating_speed(piles: list[int], hours: int) -> int:
   left, right = 1, max(piles)

   while left < right:
      speed = (left + right) // 2
      time = sum((pile + speed - 1) // speed for pile in piles)
      if time <= hours:
         right = speed
      else:
         left = speed + 1
   return left
```

**Cost:** $O(n\log m)$ time and $O(1)$ space, where $m$ is the largest pile.
