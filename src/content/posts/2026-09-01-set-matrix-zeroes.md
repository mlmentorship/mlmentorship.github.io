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
<figure class="learning-figure coding-visual-figure" aria-labelledby="set-matrix-zeroes-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="set-matrix-zeroes-state-title">Set Matrix Zeroes: Use the first row and column as markers, then apply the marked rows and columns.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="set-matrix-zeroes" role="group" aria-label="Set Matrix Zeroes: Use the first row and column as markers, then apply the marked rows and columns."><div class="coding-visual-example"><span>Input and goal</span><strong>If a cell is zero, set its full row and column to zero. Change the matrix in place.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Find zeros"><div class="coding-trace-frame-heading"><span>Find zeros</span><strong>A zero at row 1, column 1 marks its row and column.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-focus"><span>0</span><small>0</small></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span></div><div class="coding-trace-meta"><span><b>action</b>mark row 1, col 1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Read the markers"><div class="coding-trace-frame-heading"><span>Read the markers</span><strong>The first row and column now carry the future zero instructions.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state"><span>0</span><small>marker</small></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state"><span>0</span><small>marker</small></span><span class="coding-trace-grid-cell"><span>0</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Fill marked cells"><div class="coding-trace-frame-heading"><span>Fill marked cells</span><strong>Zero every cell in the marked row or column.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>0</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>0</span></span><span class="coding-trace-grid-cell"><span>0</span></span><span class="coding-trace-grid-cell"><span>0</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>0</span></span><span class="coding-trace-grid-cell"><span>1</span></span></div><div class="coding-trace-meta"><span><b>result</b>in place</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Find zeros</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Read the markers</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Fill marked cells</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Use the first row and column as markers, then apply the marked rows and columns.</p></div><figcaption><strong>Read it this way:</strong> A zero at row 1, column 1 marks its row and column. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
