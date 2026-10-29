---
title: "Unique Paths"
description: "Count disjoint final moves while one rolling row retains old-above and new-left values."
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

Count the routes across an obstacle-free rectangular grid when movement is restricted to right or down.

## Problem

**Input.** Positive integers `rows` and `cols` specifying the grid dimensions.

**Output.** Return the number of distinct move sequences from the top-left cell to the bottom-right cell.

**Rules.** Each move advances exactly one cell right or one cell down, remaining inside the grid.
There are no obstacles, diagonal moves, or backward moves.
Different move orders count separately. A one-cell grid has one route: take no moves.

**Example.** A grid with 3 rows and 2 columns returns `3`.
Its routes are right-down-down, down-right-down, and down-down-right. A 3-by-7 grid returns `28`.

## Picture

Every path enters an interior cell through exactly one final direction. The two incoming groups are disjoint, so their counts add.

Each top-row and first-column cell has exactly one route.

<!-- visual:unique-paths-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="unique-paths-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="unique-paths-state-title">Read above from the previous row and left from the row being updated</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="unique-paths" role="group" tabindex="0" aria-label="Unique Paths: Sweep a 3 by 3 grid row by row; each interior count is old ways[col] from above plus new ways[col-1] from the left."><div class="coding-visual-example"><span>Input and goal</span><strong>Count paths from the top-left to bottom-right when moves can only go right or down.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize the first row"><div class="coding-trace-frame-heading"><span>Initialize the first row</span><strong>Each top-row cell has one route; initialize ways to [1,1,1].</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="current-cell"><span>1</span><small>start/current</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>?</span></span></div><div class="coding-trace-meta"><span><b>rollingRow</b>[1,1,1]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>start/current</dt><dd><strong class="coding-state-after">(0, 0)</strong></dd></div><div class="coding-state-change"><dt>rolling row</dt><dd><strong class="coding-state-after">[1,1,1]</strong></dd></div><div class="coding-state-change"><dt>column 0 / row 0</dt><dd><strong class="coding-state-after">1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="row-one-col-one" hidden role="group" aria-label="Update row 1, column 1"><div class="coding-trace-frame-heading"><span>Update row 1, column 1</span><strong>The old ways[1] = 1 is from above and new ways[0] = 1 is from the left, so ways[1] becomes 1 + 1 = 2.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="above-dependency"><span>1</span><small>above</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="left-dependency"><span>1</span><small>left</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="current-cell"><span>2</span><small>current</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>?</span></span></div><div class="coding-trace-meta"><span><b>recurrence</b>above 1 + left 1 = 2</span><span><b>rollingRow</b>[1,2,1]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">(0, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 1)</strong></dd></div><div class="coding-state-change"><dt>above</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 1)</strong></dd></div><div class="coding-state-change"><dt>left</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 0)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="row-one-col-two" hidden role="group" aria-label="Update row 1, column 2"><div class="coding-trace-frame-heading"><span>Update row 1, column 2</span><strong>The old ways[2] = 1 is from above and ways[1] = 2 is from the left, so ways[2] becomes 1 + 2 = 3.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="above-dependency"><span>1</span><small>above</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="left-dependency"><span>2</span><small>left</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="current-cell"><span>3</span><small>current</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>?</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>?</span></span></div><div class="coding-trace-meta"><span><b>recurrence</b>above 1 + left 2 = 3</span><span><b>rollingRow</b>[1,2,3]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">(1, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 2)</strong></dd></div><div class="coding-state-change"><dt>above</dt><dd><span class="coding-state-before">(0, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 2)</strong></dd></div><div class="coding-state-change"><dt>left</dt><dd><span class="coding-state-before">(1, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 1)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="row-two-col-one" hidden role="group" aria-label="Update row 2, column 1"><div class="coding-trace-frame-heading"><span>Update row 2, column 1</span><strong>The old ways[1] = 2 is from above and the first-column value is 1, so ways[1] becomes 2 + 1 = 3.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="above-dependency"><span>2</span><small>above</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>3</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="left-dependency"><span>1</span><small>left</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="current-cell"><span>3</span><small>current</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>?</span></span></div><div class="coding-trace-meta"><span><b>recurrence</b>above 2 + left 1 = 3</span><span><b>rollingRow</b>[1,3,3]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">(1, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 1)</strong></dd></div><div class="coding-state-change"><dt>above</dt><dd><span class="coding-state-before">(0, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 1)</strong></dd></div><div class="coding-state-change"><dt>left</dt><dd><span class="coding-state-before">(1, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 0)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="destination" hidden role="group" aria-label="Reach the destination"><div class="coding-trace-frame-heading"><span>Reach the destination</span><strong>At row 2, column 2, old ways[2] = 3 comes from above and new ways[1] = 3 from the left, so 3 + 3 = 6.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>2</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="above-dependency"><span>3</span><small>above</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="left-dependency"><span>3</span><small>left</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="current-cell"><span>6</span><small>current</small></span></div><div class="coding-trace-meta"><span><b>recurrence</b>above 3 + left 3 = 6</span><span><b>rollingRow</b>[1,3,6]</span><span><b>result</b>6</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">(2, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 2)</strong></dd></div><div class="coding-state-change"><dt>above</dt><dd><span class="coding-state-before">(1, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 2)</strong></dd></div><div class="coding-state-change"><dt>left</dt><dd><span class="coding-state-before">(2, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 1)</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize the first row</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Update row 1, column 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Update row 1, column 2</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Update row 2, column 1</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Reach the destination</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when movement forms an acyclic grid and every path into a cell must arrive through a small fixed set of predecessor directions such as above and left.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">During a row sweep, ways[col] before update is the count from above and ways[col-1] after update is the count from the left; their sum is the current cell count.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When dependencies come from the previous row and current row prefix, sweep in the direction that preserves both and compress the grid to one mutable row.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Each top-row cell has one route; initialize ways to [1,1,1]. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="unique-paths-state"><strong>Check:</strong> Why is ways[column] still the value from above while ways[column-1] already belongs to the current row?</span></figcaption></figure>

## Code

Before each update, `ways[col]` still counts routes to the cell above.

`ways[col - 1]` already counts routes to the left cell in the current row. Sweep left to right to retain both meanings.

```python
def unique_paths(rows: int, cols: int) -> int:
   ways = [1] * cols
   for _ in range(1, rows):
      for col in range(1, cols):
         ways[col] += ways[col - 1]
   return ways[-1]
```

For a three-by-three grid, completed rows are `[1,1,1]`, `[1,2,3]`, then `[1,3,6]`. The answer is six.

Reversing the sweep reuses stale left counts and returns four instead.

The reference uses $O(\text{rows}\times\text{cols})$ additions and $O(\text{cols})$ integer slots. Exact counts grow, so these are not fixed-bit costs.

One row or one column gives one route. Zero or negative dimensions are outside the reference contract.

**Recall:** Which value has already advanced in time? The left slot belongs to the current row; the current slot still belongs above.

<!-- learning:core-end -->

## Block Selected Cells

**Choose this when:** Some cells cannot be entered.

Accept a rectangular grid of obstacle flags; truthy entries are blocked. Start with one arrival at the origin, then process its flag too.

A blocked cell resets its count to zero. Otherwise, the same old-above/new-left addition applies.

```python
def paths_with_obstacles(blocked):
   if not blocked or not blocked[0]:
      return 0
   ways = [0] * len(blocked[0])
   ways[0] = 1
   for row in blocked:
      for column, obstacle in enumerate(row):
         if obstacle:
            ways[column] = 0
         elif column:
            ways[column] += ways[column - 1]
   return ways[-1]
```

A three-by-three grid with its center blocked has two routes. A blocked start or destination gives zero, including a single blocked cell.

Empty grids return zero. Input rows are not modified. The algorithm uses $O(\text{rows}\times\text{columns})$ additions and $O(\text{columns})$ integer slots.

## Keep Only a Residue

**Choose this when:** The output requires the count modulo a positive integer, not the exact count.

Transposing an obstacle-free rectangle preserves the count. Allocate the shorter dimension, and reduce initialization as well as every addition.

```python
def unique_paths_mod(rows, columns, modulus):
   if modulus <= 0:
      raise ValueError("positive modulus required")
   if rows <= 0 or columns <= 0:
      return 0
   rows, columns = max(rows, columns), min(rows, columns)
   ways = [1 % modulus] * columns
   for _ in range(1, rows):
      for column in range(1, columns):
         ways[column] = (ways[column] + ways[column - 1]) % modulus
   return ways[-1]
```

Six routes modulo five give one. Modulus one gives zero even for a one-cell grid.

A nonpositive modulus raises `ValueError`. Nonpositive dimensions return zero when the modulus is valid.

The algorithm uses $O(\text{rows}\times\text{columns})$ modular additions and $O(\min(\text{rows},\text{columns}))$ residue slots. Residue arithmetic still depends on modulus bit length.

## Transfer

Unrestricted four-direction movement destroys this predecessor order. Revisiting can introduce cycles and infinitely many walks, so the same recurrence no longer specifies the task.
