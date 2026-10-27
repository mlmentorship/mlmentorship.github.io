---
title: "Reverse the edges when many starts share a few goals"
description: "Search uphill from each destination border, keep separate reachability certificates, and intersect them."
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

Find all height-grid cells from which water can reach both of two opposite ocean borders.

## Problem

**Input.** A rectangular integer height grid `heights`.

**Output.** Return zero-based `[row, col]` coordinates of every cell with a route to each ocean.

**Rules.** The Pacific touches the top and left borders; the Atlantic touches the bottom and right borders.
Water moves to equal or lower orthogonal neighbors, never diagonally or uphill.
The two routes may differ. Border cells already touch their corresponding ocean.
Preserve the grid. Empty input returns `[]`; the reference orders coordinates by row, then column.

**Example.** For this grid:

```text
1 2 2
3 2 3
2 4 1
```

Return `[[0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]]`.
This asks reachability, not water volume or travel time.

## Picture

Instead of searching from every cell, start at the destinations and walk backward along allowed edges.
If high cell `u` can flow to neighbor `v`, the reverse search at `v` can reach `u`.
Reverse movement therefore requires `neighbor_height >= current_height`.

For example, `(1,1) -> (0,1) -> (0,2)` is a forward level-height route to the Atlantic.
Its reverse `(0,2) -> (0,1) -> (1,1)` certifies the same reachability.

| Reverse search | Reached cells | Only missing cell |
| --- | --- | --- |
| Pacific | `8` | `(2,2)` |
| Atlantic | `8` | `(0,0)` |

Intersect the sets to retain exactly the cells with both certificates.

<!-- visual:pacific-atlantic-water-flow-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="pacific-atlantic-water-flow-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="pacific-atlantic-water-flow-state-title">Reverse the edges when many starts share a few goals: Reverse both flow searches from their ocean borders, move uphill, then intersect the reached cells.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="pacific-atlantic-water-flow" role="group" tabindex="0" aria-label="Reverse the edges when many starts share a few goals: Reverse both flow searches from their ocean borders, move uphill, then intersect the reached cells."><div class="coding-visual-example"><span>Input and goal</span><strong>Find cells whose water can reach both oceans.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="seed-pacific" role="group" aria-label="Seed the Pacific reverse search"><div class="coding-trace-frame-heading"><span>Seed the Pacific reverse search</span><strong>Seed the Pacific search from every top and left border cell. These five seeds initialize both seen and the stack.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-0"><span>1</span><small>P seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-1"><span>2</span><small>P seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-2"><span>2</span><small>P seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-1-0"><span>3</span><small>P seed</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>3</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-0"><span>2</span><small>P seed</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>1</span></span></div><div class="coding-trace-meta"><span><b>input</b>3x3 heights; top/left=P, bottom/right=A</span><span><b>search</b>Pacific</span><span><b>stack</b>5 border seeds</span><span><b>reached</b>5 cells</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="expand-pacific" hidden role="group" aria-label="Expand Pacific uphill"><div class="coding-trace-frame-heading"><span>Expand Pacific uphill</span><strong>Legal reverse moves add (1,1) because 2&gt;=2, (1,2) because 3&gt;=2, and (2,1) because 4&gt;=2. Height 1 at (2,2) is lower than its reached neighbors, so it is blocked.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-0"><span>1</span><small>P seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-1"><span>2</span><small>P seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-2"><span>2</span><small>P seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-1-0"><span>3</span><small>P seed</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="cell-1-1"><span>2</span><small>P via 2&lt;=2</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="cell-1-2"><span>3</span><small>P via 2&lt;=3</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-0"><span>2</span><small>P seed</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="cell-2-1"><span>4</span><small>P via 2&lt;=4</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>1</span></span></div><div class="coding-trace-meta"><span><b>input</b>3x3 heights; top/left=P, bottom/right=A</span><span><b>search</b>Pacific</span><span><b>frontier</b>(1,1),(1,2),(2,1)</span><span><b>reached</b>8 cells</span><span><b>blocked</b>(2,2): 1 is downhill</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="seed-atlantic" hidden role="group" aria-label="Seed the Atlantic reverse search"><div class="coding-trace-frame-heading"><span>Seed the Atlantic reverse search</span><strong>Atlantic starts are every bottom- or right-border cell: (0,2),(1,2),(2,0),(2,1),(2,2). The second search owns a separate seen set.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-2"><span>2</span><small>A seed</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>2</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-1-2"><span>3</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-0"><span>2</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-1"><span>4</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-2"><span>1</span><small>A seed</small></span></div><div class="coding-trace-meta"><span><b>input</b>3x3 heights; top/left=P, bottom/right=A</span><span><b>search</b>Atlantic</span><span><b>stack</b>5 border seeds</span><span><b>reached</b>5 cells</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="expand-atlantic-wave" hidden role="group" aria-label="Expand the Atlantic frontier"><div class="coding-trace-frame-heading"><span>Expand the Atlantic frontier</span><strong>From (0,2) height 2, move left to equal-height (0,1). From (2,0) height 2, move up to height 3 at (1,0).</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="cell-0-1"><span>2</span><small>A: 2&lt;=2</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-2"><span>2</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="cell-1-0"><span>3</span><small>A: 2&lt;=3</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>2</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-1-2"><span>3</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-0"><span>2</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-1"><span>4</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-2"><span>1</span><small>A seed</small></span></div><div class="coding-trace-meta"><span><b>input</b>3x3 heights; top/left=P, bottom/right=A</span><span><b>search</b>Atlantic</span><span><b>frontier</b>(0,1),(1,0)</span><span><b>reached</b>7 cells</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="finish-atlantic" hidden role="group" aria-label="Finish Atlantic reachability"><div class="coding-trace-frame-heading"><span>Finish Atlantic reachability</span><strong>From reached (0,1) height 2, move down to equal-height (1,1). Cell (0,0) height 1 is lower than 2, so reverse search cannot enter it.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-1"><span>2</span><small>A reached</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-0-2"><span>2</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-1-0"><span>3</span><small>A reached</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="cell-1-1"><span>2</span><small>A: 2&lt;=2</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-1-2"><span>3</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-0"><span>2</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-1"><span>4</span><small>A seed</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="cell-2-2"><span>1</span><small>A seed</small></span></div><div class="coding-trace-meta"><span><b>input</b>3x3 heights; top/left=P, bottom/right=A</span><span><b>search</b>Atlantic</span><span><b>frontier</b>(1,1)</span><span><b>reached</b>8 cells</span><span><b>blocked</b>(0,0): 1 is downhill</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="intersect-reached" hidden role="group" aria-label="Intersect the two reached sets"><div class="coding-trace-frame-heading"><span>Intersect the two reached sets</span><strong>Both searches reached every cell except Pacific missed (2,2) and Atlantic missed (0,0). Their intersection therefore contains the other seven coordinates.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="cell-0-1"><span>2</span><small>both</small></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="cell-0-2"><span>2</span><small>both</small></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="cell-1-0"><span>3</span><small>both</small></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="cell-1-1"><span>2</span><small>both</small></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="cell-1-2"><span>3</span><small>both</small></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="cell-2-0"><span>2</span><small>both</small></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="cell-2-1"><span>4</span><small>both</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>1</span></span></div><div class="coding-trace-meta"><span><b>input</b>3x3 heights; top/left=P, bottom/right=A</span><span><b>result</b>[[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1]]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Seed the Pacific reverse search</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Expand Pacific uphill</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Seed the Atlantic reverse search</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Expand the Atlantic frontier</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Finish Atlantic reachability</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Intersect the two reached sets</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Many grid cells ask whether they can reach either of two fixed boundary goals; reversing the edges lets each goal share one traversal across all possible starts.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Each ocean’s seen set contains exactly cells with a nonincreasing forward path to that ocean. Reverse traversal may move only to an equal-or-higher neighbor, preserving that path witness.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When many sources query reachability to a small set of goals, reverse the graph, traverse once from each goal class, and combine the resulting reachable sets.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Seed the Pacific search from every top and left border cell. These five seeds initialize both seen and the stack. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

Seed every border cell for an ocean before traversing.
Mark on discovery so each cell enters that ocean's stack once.
Keep the oceans' sets separate: one combined seed set would certify **either**, not **both**.

## Code

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

For $N=rows\times cols$, the searches use $O(N)$ time and space without changing heights.
The literal reference sorts its $K$ intersection cells.
Whole-function time is therefore $O(N+K\log(K+1))$, not unconditionally linear.
Empty or zero-width input returns `[]`.

### Failure Check

Searching downhill from the borders follows the wrong edges.
Strictly uphill search also fails: equal-height plateau routes are allowed.
One shared visited set cannot distinguish the two required destination classes.

<!-- learning:core-end -->

## Change the Goals

**New task:** Find cells that can flow to any member of an arbitrary valid goal set.

**Picture and change:** Reverse from those goals, then emit reached cells by row-major membership scan instead of sorting.
Duplicate goals are coalesced; an empty goal set reaches nothing.

```python
def reverse_reached_cells(heights, starts):
   if not heights or not heights[0]:
      return []
   rows, cols = len(heights), len(heights[0])
   seen = {tuple(cell) for cell in starts}
   stack = list(seen)
   while stack:
      row, col = stack.pop()
      for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
         next_row, next_col = row + row_step, col + col_step
         cell = next_row, next_col
         if 0 <= next_row < rows and 0 <= next_col < cols:
            if cell not in seen and heights[next_row][next_col] >= heights[row][col]:
               seen.add(cell)
               stack.append(cell)
   return [[row, col] for row in range(rows) for col in range(cols) if (row, col) in seen]
```

**Check:** `[[1,2],[3,1]]`, goal `(0,0)`, returns `[[0,0],[0,1],[1,0]]`.
The lower-right basin cannot reach that goal.
For $S$ supplied goals, time is $O(N+S)$ and auxiliary space is $O(N)$.
Heights remain unchanged.

**Use when:** Many origin cells share arbitrary destinations, including interior outlets.

## Change Both to Either

**New task:** Reaching at least one ocean is sufficient.

**Picture and change:** Combine all border goals before one reverse traversal, computing a union rather than an intersection.

```python
def reaches_either_ocean(heights):
   if not heights or not heights[0]:
      return []
   rows, cols = len(heights), len(heights[0])
   borders = (
      {(row, 0) for row in range(rows)}
      | {(row, cols - 1) for row in range(rows)}
      | {(0, col) for col in range(cols)}
      | {(rows - 1, col) for col in range(cols)}
   )
   return reverse_reached_cells(heights, borders)
```

**Check:** A `3x3` grid with border height `3` and center height `1` returns only its eight border cells.
Either-ocean reachability does not include every basin.
Time and space are $O(N)$, including deterministic row-major output.

**Use when:** The goal combination changes from AND to OR.

### Transfer Check

What changes if three goal classes must each be reachable?

### Answer

Run one reverse search per class, then intersect all three certificates.
Merging their seeds would answer a different OR question.
