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

Rotate a square matrix by ninety degrees clockwise while updating the supplied matrix in place.

## Problem

**Input.** An `n`-by-`n` integer `matrix` whose rows are distinct mutable lists; `[]` is allowed.

**Output.** Change the supplied matrix into its ninety-degree clockwise rotation and return `None`.

**Rules.** Do not return a separate rotated matrix. Values may repeat or be negative.
The input must be square, not ragged or rectangular with unequal dimensions.
External references observe changed cells and possibly reordered row objects. Empty input remains empty.

**Example.** `[[1, 2], [3, 4]]` becomes `[[3, 1], [4, 2]]`.
The original bottom-left value 3 becomes the new top-left value.

## Picture

For `[[1,2], [3,4]]`, reversing row order produces `[[3,4], [1,2]]`.
Swapping the off-diagonal pair produces `[[3,1], [4,2]]`.
Reversal sends `(row, col)` to `(n - 1 - row, col)`; transposition exchanges those coordinates.

<!-- visual:rotate-image-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="rotate-image-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="rotate-image-state-title">Rotate Image: Reverse row order, then swap every above-diagonal cell with its reflected below-diagonal partner.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="rotate-image" role="group" tabindex="0" aria-label="Rotate Image: Reverse row order, then swap every above-diagonal cell with its reflected below-diagonal partner."><div class="coding-visual-example"><span>Input and goal</span><strong>Rotate a square matrix 90 degrees clockwise in place.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Start with the original coordinates"><div class="coding-trace-frame-heading"><span>Start with the original coordinates</span><strong>The original square contains three rows: [1,2,3], [4,5,6], and [7,8,9].</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="matrix-value-1"><span>1</span><small>value 1</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-2"><span>2</span><small>value 2</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-3"><span>3</span><small>value 3</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-4"><span>4</span><small>value 4</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-5"><span>5</span><small>value 5</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-6"><span>6</span><small>value 6</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-7"><span>7</span><small>value 7</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-8"><span>8</span><small>value 8</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-9"><span>9</span><small>value 9</small></span></div><div class="coding-trace-meta"><span><b>action</b>initialize</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>matrix-value-1</dt><dd><strong class="coding-state-after">(0, 0)</strong></dd></div><div class="coding-state-change"><dt>matrix-value-2</dt><dd><strong class="coding-state-after">(0, 1)</strong></dd></div><div class="coding-state-change"><dt>matrix-value-3</dt><dd><strong class="coding-state-after">(0, 2)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="reverse-rows" hidden role="group" aria-label="Reverse the row order"><div class="coding-trace-frame-heading"><span>Reverse the row order</span><strong>matrix.reverse() moves row [7,8,9] to the top and [1,2,3] to the bottom.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-7"><span>7</span><small>move 7</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-8"><span>8</span><small>move 8</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-9"><span>9</span><small>move 9</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-4"><span>4</span><small>value 4</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-5"><span>5</span><small>value 5</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-6"><span>6</span><small>value 6</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-1"><span>1</span><small>move 1</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-2"><span>2</span><small>move 2</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-3"><span>3</span><small>move 3</small></span></div><div class="coding-trace-meta"><span><b>action</b>row 2 &lt;-&gt; row 0</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-value-7</dt><dd><span class="coding-state-before">(2, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 0)</strong></dd></div><div class="coding-state-change"><dt>matrix-value-8</dt><dd><span class="coding-state-before">(2, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 1)</strong></dd></div><div class="coding-state-change"><dt>matrix-value-9</dt><dd><span class="coding-state-before">(2, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 2)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="transpose-zero-one" hidden role="group" aria-label="Swap coordinates (0,1) and (1,0)"><div class="coding-trace-frame-heading"><span>Swap coordinates (0,1) and (1,0)</span><strong>The transpose loop first swaps 8 with 4, fixing output positions (0,1) and (1,0).</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="matrix-value-7"><span>7</span><small>value 7</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-4"><span>4</span><small>move 4</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-9"><span>9</span><small>value 9</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-8"><span>8</span><small>move 8</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-5"><span>5</span><small>value 5</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-6"><span>6</span><small>value 6</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-1"><span>1</span><small>value 1</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-2"><span>2</span><small>value 2</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-3"><span>3</span><small>value 3</small></span></div><div class="coding-trace-meta"><span><b>action</b>(0,1)=8 &lt;-&gt; (1,0)=4</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-value-4</dt><dd><span class="coding-state-before">(1, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 1)</strong></dd></div><div class="coding-state-change"><dt>matrix-value-8</dt><dd><span class="coding-state-before">(0, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 0)</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">row 2 &lt;-&gt; row 0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0,1)=8 &lt;-&gt; (1,0)=4</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="transpose-zero-two" hidden role="group" aria-label="Swap coordinates (0,2) and (2,0)"><div class="coding-trace-frame-heading"><span>Swap coordinates (0,2) and (2,0)</span><strong>Next, 9 and 1 cross the main diagonal.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="matrix-value-7"><span>7</span><small>value 7</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-4"><span>4</span><small>value 4</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-1"><span>1</span><small>move 1</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-8"><span>8</span><small>value 8</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-5"><span>5</span><small>value 5</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-6"><span>6</span><small>value 6</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-9"><span>9</span><small>move 9</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-2"><span>2</span><small>value 2</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-3"><span>3</span><small>value 3</small></span></div><div class="coding-trace-meta"><span><b>action</b>(0,2)=9 &lt;-&gt; (2,0)=1</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-value-1</dt><dd><span class="coding-state-before">(2, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 2)</strong></dd></div><div class="coding-state-change"><dt>matrix-value-9</dt><dd><span class="coding-state-before">(0, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 0)</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">(0,1)=8 &lt;-&gt; (1,0)=4</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0,2)=9 &lt;-&gt; (2,0)=1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="transpose-one-two" hidden role="group" aria-label="Swap coordinates (1,2) and (2,1)"><div class="coding-trace-frame-heading"><span>Swap coordinates (1,2) and (2,1)</span><strong>The final above-diagonal pair swaps 6 with 2; diagonal values 7,5,3 never move.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="matrix-value-7"><span>7</span><small>value 7</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-4"><span>4</span><small>value 4</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-1"><span>1</span><small>value 1</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-8"><span>8</span><small>value 8</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-5"><span>5</span><small>value 5</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-2"><span>2</span><small>move 2</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-9"><span>9</span><small>value 9</small></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="matrix-value-6"><span>6</span><small>move 6</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-3"><span>3</span><small>value 3</small></span></div><div class="coding-trace-meta"><span><b>action</b>(1,2)=6 &lt;-&gt; (2,1)=2</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>matrix-value-2</dt><dd><span class="coding-state-before">(2, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 2)</strong></dd></div><div class="coding-state-change"><dt>matrix-value-6</dt><dd><span class="coding-state-before">(1, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 1)</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">(0,2)=9 &lt;-&gt; (2,0)=1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1,2)=6 &lt;-&gt; (2,1)=2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="finish" hidden role="group" aria-label="Read the clockwise rotation"><div class="coding-trace-frame-heading"><span>Read the clockwise rotation</span><strong>Every original (r,c) now occupies (c,2-r), producing the 90-degree clockwise image in place.</strong></div><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="matrix-value-7"><span>7</span><small>value 7</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-4"><span>4</span><small>value 4</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-1"><span>1</span><small>value 1</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-8"><span>8</span><small>value 8</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-5"><span>5</span><small>value 5</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-2"><span>2</span><small>value 2</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-9"><span>9</span><small>value 9</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-6"><span>6</span><small>value 6</small></span><span class="coding-trace-grid-cell" data-motion-key="matrix-value-3"><span>3</span><small>value 3</small></span></div><div class="coding-trace-meta"><span><b>mapping</b>(r,c) -&gt; (c,2-r)</span><span><b>result</b>[[7,4,1],[8,5,2],[9,6,3]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>mapping</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(r,c) -&gt; (c,2-r)</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[[7,4,1],[8,5,2],[9,6,3]]</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">(1,2)=6 &lt;-&gt; (2,1)=2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start with the original coordinates</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Reverse the row order</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Swap coordinates (0,1) and (1,0)</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Swap coordinates (0,2) and (2,0)</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Swap coordinates (1,2) and (2,1)</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Read the clockwise rotation</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use this transformation for a square matrix that must rotate 90 degrees clockwise without allocating a second matrix.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After row reversal, values have moved from (r,c) to (n-1-r,c); after transposition, each reaches (c,n-1-r), exactly its clockwise coordinate.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Derive a target coordinate mapping, then factor it into simple reversible transforms whose in-place loops visit each swap pair exactly once.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> The original square contains three rows: [1,2,3], [4,5,6], and [7,8,9]. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="rotate-image-state"><strong>Check:</strong> Where does original cell (0,2) go after both operations? Why must each transpose pair be swapped only once?</span></figcaption></figure>

Only swap above the diagonal, where `col > row`.
Visiting both sides would swap every pair twice and undo transposition.

## Code

```python
def rotate_image(matrix: list[list[int]]) -> None:
   matrix.reverse()
   for row in range(len(matrix)):
      for col in range(row + 1, len(matrix)):
         matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
```

**Cost:** $O(n^2)$ time and $O(1)$ space.

Bounds count cell operations, not the bit lengths of stored integers.

## Change the Requirement

### Rotate Counterclockwise

**New task:** Rotate the same square matrix counterclockwise.

**Picture and change:** Transpose first, then reverse row order: `(row, col)` becomes `(n - 1 - col, row)`.

**Code:** This complete alternative retains the original mutation contract.

```python
def rotate_counterclockwise(matrix: list[list[int]]) -> None:
   for row in range(len(matrix)):
      for col in range(row + 1, len(matrix)):
         matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
   matrix.reverse()
```

**Check:** `[[1,2], [3,4]]` becomes `[[2,4], [1,3]]`.

**Use when:** Orientation reverses; costs remain $O(n^2)$ time and constant workspace.

### Preserve a Rectangular Input

**New task:** Return a rotated copy of a rectangular matrix.

**Picture and change:** An `rows`-by-`cols` input becomes `cols`-by-`rows`; allocate independent output rows.

**Code:** The input must be rectangular; empty dimensions return `[]`.

```python
def rotated_copy(matrix: list[list[int]]) -> list[list[int]]:
   rows = len(matrix)
   cols = len(matrix[0]) if rows else 0
   return [
      [matrix[row][col] for row in range(rows - 1, -1, -1)]
      for col in range(cols)
   ]
```

**Check:** `[[1,2,3], [4,5,6]]` produces `[[4,1], [5,2], [6,3]]`; the input stays unchanged.

**Use when:** Shape or ownership prevents square in-place swaps.
Time and output storage are $O(rows\times cols)$.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why exclude aliased rows?

<details>
<summary>Reasoned answer</summary>

`row = [1,2]; matrix = [row, row]` has two logical rows but only one underlying row object.
A cell update changes both logical rows, violating the coordinate-permutation model.
Build independent rows before applying the square reference.

</details>

</div>
