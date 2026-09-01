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
<figure class="learning-figure coding-visual-figure" aria-labelledby="rotting-oranges-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="rotting-oranges-state-title">Rotting Oranges: Multi-source BFS makes each queue layer one minute of spread.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="rotting-oranges" role="group" tabindex="0" aria-label="Rotting Oranges: Multi-source BFS makes each queue layer one minute of spread."><div class="coding-visual-example"><span>Input and goal</span><strong>Find how many minutes all reachable fresh oranges need to rot.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Seed all sources"><div class="coding-trace-frame-heading"><span>Seed all sources</span><strong>Every rotten orange starts in minute 0.</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">0</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">2</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-queue-item">(0,0)</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Spread one layer"><div class="coding-trace-frame-heading"><span>Spread one layer</span><strong>The minute-1 frontier reaches its fresh neighbors.</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">1</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">0</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">2</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-queue-item">(0,1)</span><span class="coding-trace-queue-item">(1,0)</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Finish at the last layer"><div class="coding-trace-frame-heading"><span>Finish at the last layer</span><strong>The final reachable orange rots at minute 4.</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">2</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">0</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">2</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">2</span><span class="coding-trace-grid-cell">2</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-empty">empty</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Seed all sources</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Spread one layer</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Finish at the last layer</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Multi-source BFS makes each queue layer one minute of spread.</p></div><figcaption><strong>Read it this way:</strong> Every rotten orange starts in minute 0. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
