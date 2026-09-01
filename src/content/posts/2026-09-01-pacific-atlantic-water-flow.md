---
title: "Pacific Atlantic Water Flow"
description: "Find cells whose water can reach both oceans."
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

> Find cells whose water can reach both oceans.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:pacific-atlantic-water-flow-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="pacific-atlantic-water-flow-state-title"><p class="visual-kicker">Reachability without repetition</p><p class="visual-title" id="pacific-atlantic-water-flow-state-title">Pacific Atlantic Water Flow: Turn a large graph into one frontier and one visited set</p><div class="coding-visual coding-visual--graph" data-coding-visual data-coding-mode="graph" data-coding-slug="pacific-atlantic-water-flow" role="group" aria-label="Pacific Atlantic Water Flow: start from both ocean borders, walk uphill, intersect reached cells. Every visited node has been scheduled exactly once, so cycles cannot repeat work."><div class="coding-visual-example"><span>Concrete trace</span><strong>start from both ocean borders, walk uphill, intersect reached cells</strong></div><div class="coding-visual-sketch coding-visual-sketch--graph"><div class="coding-sketch-graph"><span class="coding-sketch-node coding-sketch-node--active">start</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node">visited</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node coding-sketch-node--state">unseen</span></div><p class="coding-sketch-note">the frontier separates visited nodes from reachable unknowns</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Start</span><strong>current node</strong><small>Choose a source or an unseen component.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expand</span><strong>neighbors</strong><small>Follow edges or legal grid moves.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Mark</span><strong>visited</strong><small>Record a node before adding it again.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Count</span><strong>component or goal</strong><small>The explored set gives the answer.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every visited node has been scheduled exactly once, so cycles cannot repeat work.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The frontier is the boundary between known and unknown nodes. Marking a node when it enters the frontier prevents a cycle from creating duplicate searches. For this problem, hold onto the concrete trace: start from both ocean borders, walk uphill, intersect reached cells.</figcaption></figure>

**Pattern:** Reverse graph search from both goals.

**Simple idea:** Searching from every cell repeats work. Start from each ocean instead. Move
uphill or across equal height, which is the reverse of water flow. Intersect the two
reached
sets.

```python
def pacific_atlantic(heights: list[list[int]]) -> list[list[int]]:
   if not heights or not heights[0]:
      return []

   rows, cols = len(heights), len(heights[0])

   def reachable(starts: set[tuple[int, int]]) -> set[tuple[int, int]]:
      seen = set(starts)
      stack = list(starts)
      while stack:
         row, col = stack.pop()
         for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_cell = row + row_step, col + col_step
            next_row, next_col = next_cell
            if not (0 <= next_row < rows and 0 <= next_col < cols):
               continue
            if next_cell in seen or heights[next_row][next_col] < heights[row][col]:
               continue
            seen.add(next_cell)
            stack.append(next_cell)
      return seen

   pacific = {(row, 0) for row in range(rows)} | {(0, col) for col in range(cols)}
   atlantic = {(row, cols - 1) for row in range(rows)} | {
      (rows - 1, col) for col in range(cols)
   }
   return [list(cell) for cell in sorted(reachable(pacific) & reachable(atlantic))]
```

**Cost:** $O(rows \times cols)$ time and space.
