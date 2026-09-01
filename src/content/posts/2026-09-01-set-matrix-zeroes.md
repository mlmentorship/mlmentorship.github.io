---
title: "Set Matrix Zeroes"
description: "If a cell is zero, set its full row and column to zero. Change the matrix in place."
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

> If a cell is zero, set its full row and column to zero. Change the matrix in place.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:set-matrix-zeroes-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="set-matrix-zeroes-state-title"><p class="visual-kicker">Boundaries and coordinates</p><p class="visual-title" id="set-matrix-zeroes-state-title">Set Matrix Zeroes: Make the unvisited rectangle explicit before overwriting it</p><div class="coding-visual coding-visual--matrix" data-coding-visual data-coding-mode="matrix" data-coding-slug="set-matrix-zeroes" role="group" aria-label="Set Matrix Zeroes: a zero at row 1, col 2 marks the first cell of that row and column. The boundaries describe exactly which cells remain unread or unmodified."><div class="coding-visual-example"><span>Concrete trace</span><strong>a zero at row 1, col 2 marks the first cell of that row and column</strong></div><div class="coding-visual-sketch coding-visual-sketch--matrix"><div class="coding-sketch-matrix"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">focus</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">marker</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span></div><p class="coding-sketch-note">mark a row, column, layer, or active rectangle before writing over it</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Mark</span><strong>row / column</strong><small>Record information in a safe boundary or marker cell.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Visit</span><strong>current layer</strong><small>Read only the still-unvisited rectangle or ring.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Move</span><strong>boundary inward</strong><small>Shrink the region after a side is complete.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Reuse</span><strong>in-place result</strong><small>Write after the original information is safe.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The boundaries describe exactly which cells remain unread or unmodified.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The matrix becomes easier when you draw its active rectangle. Every operation either marks a future action or consumes one boundary, so no cell is accidentally read twice. For this problem, hold onto the concrete trace: a zero at row 1, col 2 marks the first cell of that row and column.</figcaption></figure>

**Pattern:** Use the first row and first column as marker storage.

**Simple idea:** Mark row `r` at `matrix[r][0]` and column `c` at `matrix[0][c]`. Save two
booleans because the first row and first column also contain real input.

```python
def _mark_zero_rows_and_cols(matrix: list[list[int]]) -> None:
   for row in range(1, len(matrix)):
      for col in range(1, len(matrix[0])):
         if matrix[row][col] == 0:
            matrix[row][0] = matrix[0][col] = 0


def _fill_marked_zeroes(matrix: list[list[int]]) -> None:
   for row in range(1, len(matrix)):
      for col in range(1, len(matrix[0])):
         if matrix[row][0] == 0 or matrix[0][col] == 0:
            matrix[row][col] = 0


def set_zeroes(matrix: list[list[int]]) -> None:
   if not matrix or not matrix[0]:
      return

   first_row_zero = 0 in matrix[0]
   first_col_zero = any(row[0] == 0 for row in matrix)
   _mark_zero_rows_and_cols(matrix)
   _fill_marked_zeroes(matrix)

   if first_row_zero:
      matrix[0] = [0] * len(matrix[0])
   if first_col_zero:
      for row in matrix:
         row[0] = 0
```

**Cost:** $O(rows \times cols)$ time and $O(1)$ extra space.
