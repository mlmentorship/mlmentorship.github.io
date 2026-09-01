---
title: "Rotate Image"
description: "Rotate a square matrix 90 degrees clockwise in place."
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

> Rotate a square matrix 90 degrees clockwise in place.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:rotate-image-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="rotate-image-state-title"><p class="visual-kicker">Boundaries and coordinates</p><p class="visual-title" id="rotate-image-state-title">Rotate Image: Make the unvisited rectangle explicit before overwriting it</p><div class="coding-visual coding-visual--matrix" data-coding-visual data-coding-mode="matrix" data-coding-slug="rotate-image" role="group" aria-label="Rotate Image: reverse rows, then transpose across the diagonal to turn columns clockwise. The boundaries describe exactly which cells remain unread or unmodified."><div class="coding-visual-example"><span>Concrete trace</span><strong>reverse rows, then transpose across the diagonal to turn columns clockwise</strong></div><div class="coding-visual-sketch coding-visual-sketch--matrix"><div class="coding-sketch-matrix"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">focus</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">marker</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span></div><p class="coding-sketch-note">mark a row, column, layer, or active rectangle before writing over it</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Mark</span><strong>row / column</strong><small>Record information in a safe boundary or marker cell.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Visit</span><strong>current layer</strong><small>Read only the still-unvisited rectangle or ring.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Move</span><strong>boundary inward</strong><small>Shrink the region after a side is complete.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Reuse</span><strong>in-place result</strong><small>Write after the original information is safe.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The boundaries describe exactly which cells remain unread or unmodified.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The matrix becomes easier when you draw its active rectangle. Every operation either marks a future action or consumes one boundary, so no cell is accidentally read twice. For this problem, hold onto the concrete trace: reverse rows, then transpose across the diagonal to turn columns clockwise.</figcaption></figure>

**Pattern:** Reverse rows, then transpose.

**Simple idea:** Reversing row order moves the bottom to the top. Swapping across the main
diagonal then puts every value in its clockwise position.

```python
def rotate_image(matrix: list[list[int]]) -> None:
   matrix.reverse()
   for row in range(len(matrix)):
      for col in range(row + 1, len(matrix)):
         matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
```

**Cost:** $O(n^2)$ time and $O(1)$ space.
