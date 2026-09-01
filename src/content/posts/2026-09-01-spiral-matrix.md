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
<figure class="learning-figure coding-visual-figure" aria-labelledby="spiral-matrix-state-title"><p class="visual-kicker">Boundaries and coordinates</p><p class="visual-title" id="spiral-matrix-state-title">Spiral Matrix: Make the unvisited rectangle explicit before overwriting it</p><div class="coding-visual coding-visual--matrix" data-coding-visual data-coding-mode="matrix" data-coding-slug="spiral-matrix" role="group" aria-label="Spiral Matrix: consume top, right, bottom, left, then shrink all four boundaries. The boundaries describe exactly which cells remain unread or unmodified."><div class="coding-visual-example"><span>Concrete trace</span><strong>consume top, right, bottom, left, then shrink all four boundaries</strong></div><div class="coding-visual-sketch coding-visual-sketch--matrix"><div class="coding-sketch-matrix"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">focus</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">marker</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span></div><p class="coding-sketch-note">mark a row, column, layer, or active rectangle before writing over it</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Mark</span><strong>row / column</strong><small>Record information in a safe boundary or marker cell.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Visit</span><strong>current layer</strong><small>Read only the still-unvisited rectangle or ring.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Move</span><strong>boundary inward</strong><small>Shrink the region after a side is complete.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Reuse</span><strong>in-place result</strong><small>Write after the original information is safe.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The boundaries describe exactly which cells remain unread or unmodified.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The matrix becomes easier when you draw its active rectangle. Every operation either marks a future action or consumes one boundary, so no cell is accidentally read twice. For this problem, hold onto the concrete trace: consume top, right, bottom, left, then shrink all four boundaries.</figcaption></figure>

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
