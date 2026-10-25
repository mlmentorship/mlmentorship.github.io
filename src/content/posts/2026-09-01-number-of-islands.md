---
title: "Count new component seeds and mark each whole region"
description: "Erase land on discovery so each component has one counted seed, then change ownership or reduce component area."
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

Count one unseen land seed, then permanently mark its entire four-neighbor component before continuing the scan.

## Problem

Count islands in this rectangular grid:

```text
1100
0101
1001
1011
```

Return **3**, not the nine land cells.
Build mutable rows with `[list(row) for row in ["1100","0101","1001","1011"]]`.
Cells are strings `"1"` and `"0"`, not integers.
Only orthogonal neighbors connect; coordinates are zero-based.
The reference consumes the grid by changing every discovered land cell to `"0"`.

## Picture

Imagine covering an entire connected patch when its first uncovered cell is found.
Later scan positions inside that patch cannot start another island.

| First unseen seed | Component cells | Running count |
| --- | --- | --- |
| `(0,0)` | `3` | `1` |
| `(1,3)` | `4` | `2` |
| `(2,0)` | `2` | `3` |

<!-- visual:number-of-islands-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="number-of-islands-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="number-of-islands-state-title">Count new component seeds and mark each whole region: Count each unseen land seed once, then erase its entire four-neighbor component with DFS.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="number-of-islands" role="group" tabindex="0" aria-label="Count new component seeds and mark each whole region: Count each unseen land seed once, then erase its entire four-neighbor component with DFS."><div class="coding-visual-example"><span>Input and goal</span><strong>Count connected groups of land in a grid.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="start-island-one" role="group" aria-label="Start island 1"><div class="coding-trace-frame-heading"><span>Start island 1</span><strong>Mark (0,0) as water before pushing it. This seed starts island 1.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="pointer-dfs"><span>0</span><small>DFS top</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-2"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-3"><span>1</span></span></div><div class="coding-trace-meta"><span><b>input</b>4x4 grid; 4-neighbor land only</span><span><b>stack</b>[(0,0)]</span><span><b>islands</b>1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="finish-island-one" hidden role="group" aria-label="Flood island 1"><div class="coding-trace-frame-heading"><span>Flood island 1</span><strong>Popping (0,0) discovers (0,1); popping (0,1) discovers (1,1). After (1,1), no connected land remains and the stack empties.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-2"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-3"><span>1</span></span></div><div class="coding-trace-meta"><span><b>input</b>4x4 grid; 4-neighbor land only</span><span><b>stack</b>empty</span><span><b>islands</b>1</span><span><b>path</b>(0,0)-&gt;(0,1)-&gt;(1,1)</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="start-island-two" hidden role="group" aria-label="Start island 2"><div class="coding-trace-frame-heading"><span>Start island 2</span><strong>The scan skips water and reaches unseen (1,3): increment islands to 2, change it to 0, and push it.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="pointer-dfs"><span>0</span><small>DFS top</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-2"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-3"><span>1</span></span></div><div class="coding-trace-meta"><span><b>input</b>4x4 grid; 4-neighbor land only</span><span><b>stack</b>[(1,3)]</span><span><b>islands</b>2</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="flood-island-two-down" hidden role="group" aria-label="Move down the second component"><div class="coding-trace-frame-heading"><span>Move down the second component</span><strong>Pop (1,3), mark neighboring (2,3) as 0 before pushing it, then pop (2,3) and push (3,3).</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-3"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-2-3"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-2"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="pointer-dfs"><span>0</span><small>DFS top</small></span></div><div class="coding-trace-meta"><span><b>input</b>4x4 grid; 4-neighbor land only</span><span><b>stack</b>[(3,3)]</span><span><b>islands</b>2</span><span><b>path</b>(1,3)-&gt;(2,3)-&gt;(3,3)</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="finish-island-two" hidden role="group" aria-label="Finish island 2"><div class="coding-trace-frame-heading"><span>Finish island 2</span><strong>Pop (3,3), discover left neighbor (3,2), and process it. The stack empties with all four cells of island 2 changed to 0.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-3"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-2-3"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-3-2"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-3-3"><span>0</span><small>visited</small></span></div><div class="coding-trace-meta"><span><b>input</b>4x4 grid; 4-neighbor land only</span><span><b>stack</b>empty</span><span><b>islands</b>2</span><span><b>path</b>(3,3)-&gt;(3,2)</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="flood-island-three" hidden role="group" aria-label="Start and flood island 3"><div class="coding-trace-frame-heading"><span>Start and flood island 3</span><strong>The scan next finds unseen (2,0), increments islands to 3, and DFS reaches its only land neighbor (3,0).</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-3"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-2-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-2-3"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="pointer-dfs"><span>0</span><small>DFS top</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-3-2"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-3-3"><span>0</span><small>visited</small></span></div><div class="coding-trace-meta"><span><b>input</b>4x4 grid; 4-neighbor land only</span><span><b>stack</b>[(3,0)]</span><span><b>islands</b>3</span><span><b>path</b>(2,0)-&gt;(3,0)</span></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="return-three" hidden role="group" aria-label="Return the component count"><div class="coding-trace-frame-heading"><span>Return the component count</span><strong>After (3,0) is processed, the stack and remaining scan contain no unseen land. Exactly three DFS starts occurred.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-0-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-1"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-1-3"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-2-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-2-3"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-3-0"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-3-2"><span>0</span><small>visited</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="visited-3-3"><span>0</span><small>visited</small></span></div><div class="coding-trace-meta"><span><b>input</b>4x4 grid; 4-neighbor land only</span><span><b>stack</b>empty</span><span><b>result</b>3 islands</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 7</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start island 1</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Flood island 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Start island 2</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Move down the second component</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Finish island 2</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Start and flood island 3</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Return the component count</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">The task asks for connected groups in a binary grid under four-direction adjacency, so every unseen land cell identifies exactly one not-yet-counted component.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before the scan advances, every visited land cell has been changed to 0; each stack contains only cells from the current island, and each completed island can never be counted again.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For regions, blobs, and connected-component problems, scan for unseen seeds, count seeds rather than cells, and exhaustively mark each seed’s reachable component before continuing.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Mark (0,0) as water before pushing it. This seed starts island 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

Mark a seed before placing it on the stack.
Mark each new neighbor before pushing it, too.
Every land cell is therefore scheduled once, even when several neighbors could discover it.
Stack order changes visitation order, not component membership.

## Code

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

Scanning and coverage take $O(rows\times cols)$ time.
The explicit stack can require $O(rows\times cols)$ auxiliary space.
There is no logarithmic-stack guarantee.
Empty and zero-width grids return `0`.

### Failure Check

The diagonal grid `["10","01"]` has two islands under this adjacency rule.
Counting each popped cell returns area, not island count.
Restoring marks would allow later scan positions to count the same component again.

<!-- learning:core-end -->

## Change the Ownership

**New task:** Count islands without changing the caller's grid.

**Picture and change:** Copy each row before destructive coverage.
The wrapper also accepts string rows by converting them to mutable character lists.

```python
def islands_preserving(grid):
   return num_islands([list(row) for row in grid])
```

**Check:** The source still returns `3`, with all original cells unchanged.
The copy adds $O(rows\times cols)$ storage.

**Use when:** Mutation permission changes, not connectivity.

## Change the Reduction

**New task:** Return the largest island area.

**Picture and change:** Keep permanent discovery marks, count cells within each completed flood, then maximize those counts.
This version still consumes mutable string-cell rows.

```python
def largest_island_area(grid):
   best = 0
   for row in range(len(grid)):
      for col in range(len(grid[0])):
         if grid[row][col] != "1":
            continue
         grid[row][col] = "0"
         stack, area = [(row, col)], 0
         while stack:
            current_row, current_col = stack.pop()
            area += 1
            for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
               next_row, next_col = current_row + row_step, current_col + col_step
               if 0 <= next_row < len(grid) and 0 <= next_col < len(grid[0]):
                  if grid[next_row][next_col] == "1":
                     grid[next_row][next_col] = "0"
                     stack.append((next_row, next_col))
         best = max(best, area)
   return best
```

**Check:** Source component areas are `3,4,2`, so return `4`.
No land returns `0`.
Time and worst-case stack bounds remain unchanged.

**Use when:** Component size, not the number of components, is requested.

### Transfer Check

Why keep marks permanently here but undo used-cell marks in a word-search branch?

### Answer

Component membership is a global completed fact.
A candidate path's used cells constrain only that path; other paths must be allowed to try them again.
Compare [reversed reachability](/questions/pacific-atlantic-water-flow/) when the task asks which goals each cell can reach.
