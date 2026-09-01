---
title: "Rotting Oranges"
description: "Find how many minutes all reachable fresh oranges need to rot."
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

> Find how many minutes all reachable fresh oranges need to rot.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:rotting-oranges-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="rotting-oranges-state-title"><p class="visual-kicker">Distance in layers</p><p class="visual-title" id="rotting-oranges-state-title">Rotting Oranges: A queue turns time or steps into visible layers</p><div class="coding-visual coding-visual--bfs" data-coding-visual data-coding-mode="bfs" data-coding-slug="rotting-oranges" role="group" aria-label="Rotting Oranges: all rotten cells seed minute 0; each queue layer is one minute. The queue is ordered by nondecreasing distance from the starting frontier."><div class="coding-visual-example"><span>Concrete trace</span><strong>all rotten cells seed minute 0; each queue layer is one minute</strong></div><div class="coding-visual-sketch coding-visual-sketch--bfs"><div class="coding-sketch-grid"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--seen">0</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--frontier">1</span><span class="coding-sketch-grid-cell">2</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--seen">1</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--frontier">2</span><span class="coding-sketch-grid-cell">3</span><span class="coding-sketch-grid-cell">2</span><span class="coding-sketch-grid-cell">3</span><span class="coding-sketch-grid-cell">4</span></div><p class="coding-sketch-note">each layer is one more step or minute from the starting frontier</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Seed</span><strong>frontier at 0</strong><small>Put every starting position in the queue.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Pop</span><strong>current layer</strong><small>Process only positions at the same distance.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Spread</span><strong>next layer</strong><small>Add each newly reachable neighbor once.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Finish</span><strong>first arrival</strong><small>The first layer reaching a goal is shortest.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The queue is ordered by nondecreasing distance from the starting frontier.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read each queue layer as one minute or one step. Multiple starting points belong in the first layer, which is why multi-source BFS measures the nearest source. For this problem, hold onto the concrete trace: all rotten cells seed minute 0; each queue layer is one minute.</figcaption></figure>

**Pattern:** Multi-source BFS.

**Simple idea:** Every rotten orange starts spreading at time 0. Put all of them in the queue
before BFS. Each queue level is one minute.

```python
from collections import deque

def rotting_oranges(grid: list[list[int]]) -> int:
   if not grid or not grid[0]:
      return 0

   queue = deque(
      (row, col, 0)
      for row in range(len(grid))
      for col in range(len(grid[0]))
      if grid[row][col] == 2
   )
   fresh = sum(cell == 1 for row in grid for cell in row)

   minutes = 0
   while queue:
      row, col, minutes = queue.popleft()
      for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
         new_row = row + row_step
         new_col = col + col_step
         if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
            if grid[new_row][new_col] == 1:
               grid[new_row][new_col] = 2
               fresh -= 1
               queue.append((new_row, new_col, minutes + 1))

   return minutes if fresh == 0 else -1
```

**Cost:** $O(rows \times cols)$ time and space.
