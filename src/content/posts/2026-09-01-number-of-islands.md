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
<figure class="learning-figure coding-visual-figure" aria-labelledby="number-of-islands-state-title"><p class="visual-kicker">Reachability without repetition</p><p class="visual-title" id="number-of-islands-state-title">Number of Islands: Turn a large graph into one frontier and one visited set</p><div class="coding-visual coding-visual--graph" data-coding-visual data-coding-mode="graph" data-coding-slug="number-of-islands" role="group" aria-label="Number of Islands: each unseen 1 starts one flood; mark its whole connected land as 0. Every visited node has been scheduled exactly once, so cycles cannot repeat work."><div class="coding-visual-example"><span>Concrete trace</span><strong>each unseen 1 starts one flood; mark its whole connected land as 0</strong></div><div class="coding-visual-sketch coding-visual-sketch--graph"><div class="coding-sketch-graph"><span class="coding-sketch-node coding-sketch-node--active">start</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node">visited</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node coding-sketch-node--state">unseen</span></div><p class="coding-sketch-note">the frontier separates visited nodes from reachable unknowns</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Start</span><strong>current node</strong><small>Choose a source or an unseen component.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expand</span><strong>neighbors</strong><small>Follow edges or legal grid moves.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Mark</span><strong>visited</strong><small>Record a node before adding it again.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Count</span><strong>component or goal</strong><small>The explored set gives the answer.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every visited node has been scheduled exactly once, so cycles cannot repeat work.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The frontier is the boundary between known and unknown nodes. Marking a node when it enters the frontier prevents a cycle from creating duplicate searches. For this problem, hold onto the concrete trace: each unseen 1 starts one flood; mark its whole connected land as 0.</figcaption></figure>

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
