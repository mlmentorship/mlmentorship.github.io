---
title: "Spiral Matrix"
description: "Return matrix values in spiral order."
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

> Return matrix values in spiral order.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:spiral-matrix-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="spiral-matrix-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="spiral-matrix-state-title">Spiral Matrix: Read the four current boundaries, then shrink them after each side.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="spiral-matrix" role="group" tabindex="0" aria-label="Spiral Matrix: Read the four current boundaries, then shrink them after each side."><div class="coding-visual-example"><span>Input and goal</span><strong>Return matrix values in spiral order.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Read the top and right"><div class="coding-trace-frame-heading"><span>Read the top and right</span><strong>Consume top row 1,2,3 and right column 6,9.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="marker-top"><span>1</span><small>top</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>8</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="marker-right"><span>9</span><small>right</small></span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Read bottom and left"><div class="coding-trace-frame-heading"><span>Read bottom and left</span><strong>Continue backward across 8,7 and up through 4.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>.</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="marker-left"><span>4</span><small>left</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>7</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="marker-bottom"><span>8</span><small>bottom</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>.</span></span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Finish the inner layer"><div class="coding-trace-frame-heading"><span>Finish the inner layer</span><strong>The remaining center is 5.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>.</span></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="marker-last"><span>5</span><small>last</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>.</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>.</span></span></div><div class="coding-trace-meta"><span><b>result</b>[1,2,3,6,9,8,7,4,5]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Read the top and right</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Read bottom and left</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Finish the inner layer</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Read the four current boundaries, then shrink them after each side.</p></div><figcaption><strong>Read it this way:</strong> Consume top row 1,2,3 and right column 6,9. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Shrinking top, bottom, left, and right boundaries.

**Simple idea:** Read the top row, right column, bottom row, and left column. Move each used
boundary inward. Check that a row or column still exists before reading it.

```python
def spiral_order(matrix: list[list[int]]) -> list[int]:
   if not matrix or not matrix[0]:
      return []

   answer = []
   top, bottom = 0, len(matrix) - 1
   left, right = 0, len(matrix[0]) - 1

   while top <= bottom and left <= right:
      answer.extend(matrix[top][left : right + 1])
      top += 1

      for row in range(top, bottom + 1):
         answer.append(matrix[row][right])
      right -= 1

      if top <= bottom:
         answer.extend(reversed(matrix[bottom][left : right + 1]))
         bottom -= 1
      if left <= right:
         for row in range(bottom, top - 1, -1):
            answer.append(matrix[row][left])
         left += 1

   return answer
```

**Cost:** $O(rows \times cols)$ time and $O(1)$ extra space, not counting the answer.
