---
title: "Number of Islands"
description: "Count connected groups of land in a grid."
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

> Count connected groups of land in a grid.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:number-of-islands-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="number-of-islands-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="number-of-islands-state-title">Number of Islands: Start a flood only at unseen land, then mark the whole component.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="number-of-islands" role="group" aria-label="Number of Islands: Start a flood only at unseen land, then mark the whole component."><div class="coding-visual-example"><span>Input and goal</span><strong>Count connected groups of land in a grid.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Find the first land"><div class="coding-trace-frame-heading"><span>Find the first land</span><strong>The top-left 1 starts island 1.</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">1</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">2</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">1</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">3</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-queue-item">(0,0)</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Flood the component"><div class="coding-trace-frame-heading"><span>Flood the component</span><strong>Every connected 1 becomes visited water 0.</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">2</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">3</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-empty">empty</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Count starts, not cells"><div class="coding-trace-frame-heading"><span>Count starts, not cells</span><strong>Only the first unseen land cell increments the island count.</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span><span class="coding-trace-grid-cell">0</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-empty">empty</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Find the first land</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Flood the component</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Count starts, not cells</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Start a flood only at unseen land, then mark the whole component.</p></div><figcaption><strong>Read it this way:</strong> The top-left 1 starts island 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** DFS from each unseen land cell.

**Simple idea:** Each unseen land cell starts one new island. DFS changes every connected
land cell to water, so that island is never counted again.

```python
def num_islands(grid: list[list[str]]) -> int:
   islands = 0

   for row in range(len(grid)):
      for col in range(len(grid[0])):
         if grid[row][col] != "1":
            continue

         islands += 1
         grid[row][col] = "0"
         stack = [(row, col)]
         while stack:
            current_row, current_col = stack.pop()
            for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
               new_row = current_row + row_step
               new_col = current_col + col_step
               if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                  if grid[new_row][new_col] == "1":
                     grid[new_row][new_col] = "0"
                     stack.append((new_row, new_col))

   return islands
```

**Cost:** $O(rows \times cols)$ time and space.
