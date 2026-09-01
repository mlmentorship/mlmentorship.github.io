---
title: "Word Search"
description: "Check whether a word can be formed by neighboring board cells without reusing a cell."
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

> Check whether a word can be formed by neighboring board cells without reusing a cell.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:word-search-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="word-search-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="word-search-state-title">Word Search: Mark a board cell while it belongs to the current path, then restore it on return.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="word-search" role="group" aria-label="Word Search: Mark a board cell while it belongs to the current path, then restore it on return."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether a word can be formed by neighboring board cells without reusing a cell.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Start at A"><div class="coding-trace-frame-heading"><span>Start at A</span><strong>The first matching cell starts the path A.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus"><span>A</span><small>A</small></span><span class="coding-trace-grid-cell"><span>B</span></span><span class="coding-trace-grid-cell"><span>C</span></span><span class="coding-trace-grid-cell"><span>E</span></span><span class="coding-trace-grid-cell"><span>S</span></span><span class="coding-trace-grid-cell"><span>F</span></span><span class="coding-trace-grid-cell"><span>C</span></span><span class="coding-trace-grid-cell"><span>S</span></span><span class="coding-trace-grid-cell"><span>A</span></span><span class="coding-trace-grid-cell"><span>D</span></span><span class="coding-trace-grid-cell"><span>E</span></span><span class="coding-trace-grid-cell"><span>E</span></span></div><div class="coding-trace-meta"><span><b>word</b>A B C C E D</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Extend the path"><div class="coding-trace-frame-heading"><span>Extend the path</span><strong>Move through adjacent B, C, and C cells while marking them used.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state"><span>#</span><small>A</small></span><span class="coding-trace-grid-cell trace-tone-state"><span>#</span><small>B</small></span><span class="coding-trace-grid-cell trace-tone-state"><span>#</span><small>C</small></span><span class="coding-trace-grid-cell"><span>E</span></span><span class="coding-trace-grid-cell"><span>S</span></span><span class="coding-trace-grid-cell"><span>F</span></span><span class="coding-trace-grid-cell trace-tone-focus"><span>#</span><small>C</small></span><span class="coding-trace-grid-cell"><span>S</span></span><span class="coding-trace-grid-cell"><span>A</span></span><span class="coding-trace-grid-cell"><span>D</span></span><span class="coding-trace-grid-cell"><span>E</span></span><span class="coding-trace-grid-cell"><span>E</span></span></div><div class="coding-trace-meta"><span><b>word</b>A -&gt; B -&gt; C -&gt; C</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Reach the final D"><div class="coding-trace-frame-heading"><span>Reach the final D</span><strong>Continue to E and D; restore cells when a branch fails.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell"><span>#</span></span><span class="coding-trace-grid-cell"><span>#</span></span><span class="coding-trace-grid-cell"><span>#</span></span><span class="coding-trace-grid-cell"><span>E</span></span><span class="coding-trace-grid-cell"><span>S</span></span><span class="coding-trace-grid-cell"><span>F</span></span><span class="coding-trace-grid-cell"><span>#</span></span><span class="coding-trace-grid-cell"><span>S</span></span><span class="coding-trace-grid-cell"><span>A</span></span><span class="coding-trace-grid-cell trace-tone-output"><span>D</span><small>D</small></span><span class="coding-trace-grid-cell"><span>#</span></span><span class="coding-trace-grid-cell"><span>E</span></span></div><div class="coding-trace-meta"><span><b>result</b>ABCCED found</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start at A</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Extend the path</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Reach the final D</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Mark a board cell while it belongs to the current path, then restore it on return.</p></div><figcaption><strong>Read it this way:</strong> The first matching cell starts the path A. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Backtracking on a grid.

**Simple idea:** Start from each cell. Temporarily mark a chosen cell so the current path
cannot use it again. Restore it before returning.

```python
def word_search(board: list[list[str]], word: str) -> bool:
   if not word:
      return True
   if not board or not board[0]:
      return False

   def search(row: int, col: int, index: int) -> bool:
      if index == len(word):
         return True
      if not (0 <= row < len(board) and 0 <= col < len(board[0])):
         return False
      if board[row][col] != word[index]:
         return False

      char = board[row][col]
      board[row][col] = "#"
      found = (
         search(row + 1, col, index + 1)
         or search(row - 1, col, index + 1)
         or search(row, col + 1, index + 1)
         or search(row, col - 1, index + 1)
      )
      board[row][col] = char
      return found

   return any(
      search(row, col, 0)
      for row in range(len(board))
      for col in range(len(board[0]))
   )
```

**Cost:** $O(rows \times cols \times 4^L)$ time and $O(L)$ space, where $L$ is word length.
