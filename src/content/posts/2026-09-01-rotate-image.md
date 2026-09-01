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
<figure class="learning-figure coding-visual-figure" aria-labelledby="rotate-image-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="rotate-image-state-title">Rotate Image: Reverse the row order, then transpose across the main diagonal.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="rotate-image" role="group" tabindex="0" aria-label="Rotate Image: Reverse the row order, then transpose across the main diagonal."><div class="coding-visual-example"><span>Input and goal</span><strong>Rotate a square matrix 90 degrees clockwise in place.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Reverse rows"><div class="coding-trace-frame-heading"><span>Reverse rows</span><strong>The bottom row moves to the top.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>3</span></span></div><div class="coding-trace-meta"><span><b>action</b>reverse rows</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Transpose"><div class="coding-trace-frame-heading"><span>Transpose</span><strong>Swap cells across the diagonal: (row,col) becomes (col,row).</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="marker-fixed"><span>7</span><small>fixed</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="marker-moved"><span>1</span><small>moved</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>3</span></span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Read clockwise result"><div class="coding-trace-frame-heading"><span>Read clockwise result</span><strong>The matrix is rotated in place without a second matrix.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>3</span></span></div><div class="coding-trace-meta"><span><b>result</b>90 degrees clockwise</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Reverse rows</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Transpose</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read clockwise result</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Reverse the row order, then transpose across the main diagonal.</p></div><figcaption><strong>Read it this way:</strong> The bottom row moves to the top. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
