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
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-increasing-path-in-a-matrix-state-title"><p class="visual-kicker">Boundaries and coordinates</p><p class="visual-title" id="longest-increasing-path-in-a-matrix-state-title">Longest Increasing Path in a Matrix: Make the unvisited rectangle explicit before overwriting it</p><div class="coding-visual coding-visual--matrix" data-coding-visual data-coding-mode="matrix" data-coding-slug="longest-increasing-path-in-a-matrix" role="group" aria-label="Longest Increasing Path in a Matrix: cache the best path from each cell; larger-only moves cannot cycle. The boundaries describe exactly which cells remain unread or unmodified."><div class="coding-visual-example"><span>Concrete trace</span><strong>cache the best path from each cell; larger-only moves cannot cycle</strong></div><div class="coding-visual-sketch coding-visual-sketch--matrix"><div class="coding-sketch-matrix"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">focus</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">marker</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span></div><p class="coding-sketch-note">mark a row, column, layer, or active rectangle before writing over it</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Mark</span><strong>row / column</strong><small>Record information in a safe boundary or marker cell.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Visit</span><strong>current layer</strong><small>Read only the still-unvisited rectangle or ring.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Move</span><strong>boundary inward</strong><small>Shrink the region after a side is complete.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Reuse</span><strong>in-place result</strong><small>Write after the original information is safe.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The boundaries describe exactly which cells remain unread or unmodified.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The matrix becomes easier when you draw its active rectangle. Every operation either marks a future action or consumes one boundary, so no cell is accidentally read twice. For this problem, hold onto the concrete trace: cache the best path from each cell; larger-only moves cannot cycle.</figcaption></figure>

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
