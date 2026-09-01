---
title: "Longest Increasing Path in a Matrix"
description: "Find the longest path that moves to a larger neighboring value each step."
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

> Find the longest path that moves to a larger neighboring value each step.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-increasing-path-in-a-matrix-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-increasing-path-in-a-matrix-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-increasing-path-in-a-matrix-state-title">Longest Increasing Path in a Matrix: Memoize the best increasing path starting at each cell; larger-only moves cannot cycle.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-increasing-path-in-a-matrix" role="group" tabindex="0" aria-label="Longest Increasing Path in a Matrix: Memoize the best increasing path starting at each cell; larger-only moves cannot cycle."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the longest path that moves to a larger neighboring value each step.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Find increasing neighbors"><div class="coding-trace-frame-heading"><span>Find increasing neighbors</span><strong>From 1, move to 2, then 6, then 9.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>8</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="marker-2"><span>2</span><small>2</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="marker-1"><span>1</span><small>1</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>1</span></span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Cache a cell answer"><div class="coding-trace-frame-heading"><span>Cache a cell answer</span><strong>The memo table stores the best path length from every cell; the path from 1 has length 4.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>3</span></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="marker-path-length-4"><span>4</span><small>path length 4</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>3</span></span></div><div class="coding-trace-meta"><span><b>action</b>memoize</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Take the maximum cached value"><div class="coding-trace-frame-heading"><span>Take the maximum cached value</span><strong>Every cell is solved once; the largest cached path is 4.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>3</span></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="marker-max-4"><span>4</span><small>max 4</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>3</span></span></div><div class="coding-trace-meta"><span><b>result</b>4</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Find increasing neighbors</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Cache a cell answer</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Take the maximum cached value</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Memoize the best increasing path starting at each cell; larger-only moves cannot cycle.</p></div><figcaption><strong>Read it this way:</strong> From 1, move to 2, then 6, then 9. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** DFS plus memoization.

**Simple idea:** The longest path from one cell never changes. Cache it. Larger-only moves
also prevent cycles.

```python
from functools import cache

def longest_increasing_path(matrix: list[list[int]]) -> int:
   if not matrix or not matrix[0]:
      return 0

   @cache
   def path_from(row: int, col: int) -> int:
      best = 1
      for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
         new_row = row + row_step
         new_col = col + col_step
         if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
            if matrix[new_row][new_col] > matrix[row][col]:
               best = max(best, 1 + path_from(new_row, new_col))
      return best

   return max(
      path_from(row, col)
      for row in range(len(matrix))
      for col in range(len(matrix[0]))
   )
```

**Cost:** $O(rows \times cols)$ time and space.
