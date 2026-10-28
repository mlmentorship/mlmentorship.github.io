---
title: "Set Matrix Zeroes"
description: "If a cell is zero, set its full row and column to zero. Change the matrix in place."
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

Update a matrix so every row and column containing an original zero becomes entirely zero.

## Problem

**Input.** A rectangular integer `matrix` whose rows are distinct mutable lists.

**Output.** Modify the matrix in place and return `None`.
Every row or column containing a zero in the original matrix must become all zeros.

**Rules.** Newly written zeros must not trigger further propagation beyond those original rows and columns.
Other cells retain their original values. Negative and repeated nonzero values are allowed.
Empty dimensions do nothing. The reference may replace the first row object as well as changing cells.

**Example.** `[[1, 1, 1], [1, 0, 1], [1, 1, 1]]` becomes `[[1, 0, 1], [0, 0, 0], [1, 0, 1]]`.
The four corner cells stay 1 because their original rows and columns contained no zero.

## Picture

In `[[1,0,3], [4,5,6], [7,8,0]]`, original zeros require rows `0,2` and columns `1,2`.
Save whether the first row and column originally contained zeros.
Then mark interior discoveries at `matrix[row][0]` and `matrix[0][col]`.

<!-- visual:set-matrix-zeroes-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="set-matrix-zeroes-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="set-matrix-zeroes-state-title">Set Matrix Zeroes: Preserve first-row and first-column facts, store interior zero instructions in those borders, then apply markers before restoring the borders.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="set-matrix-zeroes" role="group" tabindex="0" aria-label="Set Matrix Zeroes: Preserve first-row and first-column facts, store interior zero instructions in those borders, then apply markers before restoring the borders."><div class="coding-visual-example"><span>Input and goal</span><strong>If a cell is zero, set its full row and column to zero. Change the matrix in place.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="inspect-borders" role="group" aria-label="Inspect the original borders"><div class="coding-trace-frame-heading"><span>Inspect the original borders</span><strong>The first row contains zero. The first column does not.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-cursor"><span>0</span><small>scan</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span></div><div class="coding-trace-meta"><span><b>coordinates</b>row,column</span><span><b>firstRowZero</b>true</span><span><b>firstColZero</b>false</span><span><b>action</b>save border facts before overwriting markers</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>matrix-cursor</dt><dd><strong class="coding-state-after">(0, 1)</strong></dd></div><div class="coding-state-change"><dt>coordinates</dt><dd><strong class="coding-state-after">row,column</strong></dd></div><div class="coding-state-change"><dt>first row zero</dt><dd><strong class="coding-state-after">true</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="scan-row-one" hidden role="group" aria-label="Scan the first interior row"><div class="coding-trace-frame-heading"><span>Scan the first interior row</span><strong>Cells (1,1)=5 and (1,2)=6 are nonzero, so row marker (1,0)=4 is unchanged.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>5</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-cursor"><span>6</span><small>scan</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span></div><div class="coding-trace-meta"><span><b>coordinates</b>row,column</span><span><b>firstRowZero</b>true</span><span><b>firstColZero</b>false</span><span><b>action</b>no interior marker written in row 1</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-cursor</dt><dd><span class="coding-state-before">(0, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 2)</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">save border facts before overwriting markers</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">no interior marker written in row 1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="mark-row-two-col-two" hidden role="group" aria-label="Mark the interior zero"><div class="coding-trace-frame-heading"><span>Mark the interior zero</span><strong>Cell (2,2)=0 writes row marker matrix[2][0]=0 and column marker matrix[0][2]=0.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>8</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-cursor"><span>0</span><small>scan</small></span></div><div class="coding-trace-meta"><span><b>coordinates</b>row,column</span><span><b>action</b>(2,2)=0 -&gt; write markers (2,0) and (0,2)</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-cursor</dt><dd><span class="coding-state-before">(1, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 2)</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">no interior marker written in row 1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2,2)=0 -&gt; write markers (2,0) and (0,2)</strong></dd></div><div class="coding-state-change"><dt>first row zero</dt><dd><span class="coding-state-before">true</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="fill-row-one" hidden role="group" aria-label="Fill row 1 from column markers"><div class="coding-trace-frame-heading"><span>Fill row 1 from column markers</span><strong>At (1,1), matrix[0][1]=0; at (1,2), matrix[0][2]=0. Both interior cells become 0.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-cursor"><span>0</span><small>scan</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span></div><div class="coding-trace-meta"><span><b>coordinates</b>row,column</span><span><b>action</b>column markers zero (1,1) and (1,2)</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-cursor</dt><dd><span class="coding-state-before">(2, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 2)</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">(2,2)=0 -&gt; write markers (2,0) and (0,2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">column markers zero (1,1) and (1,2)</strong></dd></div><div class="coding-state-change"><dt>column 1 / row 1</dt><dd><span class="coding-state-before">5</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="fill-row-two" hidden role="group" aria-label="Fill row 2 from its row marker"><div class="coding-trace-frame-heading"><span>Fill row 2 from its row marker</span><strong>matrix[2][0]=0, so (2,1) and (2,2) are zeroed regardless of their column markers.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>0</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-cursor"><span>0</span><small>scan</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span></div><div class="coding-trace-meta"><span><b>coordinates</b>row,column</span><span><b>action</b>row marker zeroes row 2 interior</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-cursor</dt><dd><span class="coding-state-before">(1, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 1)</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">column markers zero (1,1) and (1,2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">row marker zeroes row 2 interior</strong></dd></div><div class="coding-state-change"><dt>column 1 / row 2</dt><dd><span class="coding-state-before">8</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="restore-borders" hidden role="group" aria-label="Restore the first row"><div class="coding-trace-frame-heading"><span>Restore the first row</span><strong>first_row_zero is true, so replace row 0 with [0,0,0]. first_col_zero is false, so no separate column pass runs.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-cursor"><span>0</span><small>scan</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>0</span></span></div><div class="coding-trace-meta"><span><b>coordinates</b>row,column</span><span><b>action</b>zero first row; skip first-column pass</span><span><b>result</b>[[0,0,0],[4,0,0],[0,0,0]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-cursor</dt><dd><span class="coding-state-before">(2, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 0)</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">row marker zeroes row 2 interior</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">zero first row; skip first-column pass</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[[0,0,0],[4,0,0],[0,0,0]]</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Inspect the original borders</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Scan the first interior row</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Mark the interior zero</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Fill row 1 from column markers</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Fill row 2 from its row marker</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Restore the first row</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use border markers when entire rows and columns must change from discovered cells, but the required extra space must remain constant.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After the marker pass, matrix[r][0]=0 means interior row r must be zeroed and matrix[0][c]=0 means interior column c must be zeroed; saved booleans independently preserve the original first-border obligations.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When output overwrites input, identify safe in-place metadata cells and separately save any original facts those cells represented before reusing them.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> The first row contains zero. The first column does not. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="set-matrix-zeroes-state"><strong>Check:</strong> Why does a marker zero not imply that the original first column was entirely zero?</span></figcaption></figure>

Complete discovery before filling the interior, so newly written zeros never become new discoveries.
Finally restore the border obligations saved before marking.
The result is `[[0,0,0], [4,0,0], [0,0,0]]`.

## Code

```python
def _mark_zero_rows_and_cols(matrix: list[list[int]]) -> None:
   for row in range(1, len(matrix)):
      for col in range(1, len(matrix[0])):
         if matrix[row][col] == 0:
            matrix[row][0] = matrix[0][col] = 0


def _fill_marked_zeroes(matrix: list[list[int]]) -> None:
   for row in range(1, len(matrix)):
      for col in range(1, len(matrix[0])):
         if matrix[row][0] == 0 or matrix[0][col] == 0:
            matrix[row][col] = 0


def set_zeroes(matrix: list[list[int]]) -> None:
   if not matrix or not matrix[0]:
      return

   first_row_zero = 0 in matrix[0]
   first_col_zero = any(row[0] == 0 for row in matrix)
   _mark_zero_rows_and_cols(matrix)
   _fill_marked_zeroes(matrix)

   if first_row_zero:
      matrix[0] = [0] * len(matrix[0])
   if first_col_zero:
      for row in matrix:
         row[0] = 0
```

**Cost:** $O(rows\times cols)$ cell operations and $O(cols)$ worst-case temporary storage for first-row replacement.
Marker state itself is constant-sized.

## Change the Requirement

### Preserve Every Row Object

**New task:** Keep row identities and use constant workspace.

**Picture and change:** Write border zeros cellwise instead of allocating a replacement row.

**Code:** Reuse the two complete helpers above.

```python
def set_zeroes_preserving_rows(matrix: list[list[int]]) -> None:
   if not matrix or not matrix[0]:
      return
   first_row_zero = 0 in matrix[0]
   first_col_zero = any(row[0] == 0 for row in matrix)
   _mark_zero_rows_and_cols(matrix)
   _fill_marked_zeroes(matrix)
   if first_row_zero:
      for col in range(len(matrix[0])):
         matrix[0][col] = 0
   if first_col_zero:
      for row in matrix:
         row[0] = 0
```

**Check:** `[[1,0], [3,4]]` becomes `[[0,0], [3,0]]`, with both original row objects retained.

**Use when:** External row references must observe the full mutation.
Time remains linear in cells; workspace becomes constant.

### Leave the Input Unchanged

**New task:** Return an independent result.

**Picture and change:** Copy each row before borrowing its border cells.

**Code:** Call the complete row-preserving implementation above.

```python
def zeroed_copy(matrix: list[list[int]]) -> list[list[int]]:
   result = [row[:] for row in matrix]
   set_zeroes_preserving_rows(result)
   return result
```

**Check:** `zeroed_copy([[1,0], [3,4]])` returns `[[0,0], [3,0]]` without changing the source.

**Use when:** Callers retain an immutable input snapshot.
Output storage is $O(rows\times cols)$.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why not zero a row immediately upon discovery?

<details>
<summary>Reasoned answer</summary>

For `[[1,0], [1,1]]`, immediate filling creates a new zero at `(0,0)`.
Treating that write as original evidence incorrectly clears column zero too.
Separate discovery from application, and preserve original border facts before using borders as instructions.

</details>

</div>
