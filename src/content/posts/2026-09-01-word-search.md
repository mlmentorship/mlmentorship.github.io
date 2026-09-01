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
<figure class="learning-figure coding-visual-figure" aria-labelledby="word-search-state-title"><p class="visual-kicker">A tree of choices</p><p class="visual-title" id="word-search-state-title">Word Search: Choose, explore, then undo the exact choice</p><div class="coding-visual coding-visual--backtrack" data-coding-visual data-coding-mode="backtrack" data-coding-slug="word-search" role="group" aria-label="Word Search: trace C-A-T through neighboring cells, marking each chosen cell temporarily. At every call, the path contains exactly the choices on the route from the root."><div class="coding-visual-example"><span>Concrete trace</span><strong>trace C-A-T through neighboring cells, marking each chosen cell temporarily</strong></div><div class="coding-visual-sketch coding-visual-sketch--backtrack"><div class="coding-sketch-choice-tree"><span class="coding-sketch-node coding-sketch-node--active">partial path</span><div class="coding-sketch-choice-branches"><span class="coding-sketch-node">choose A</span><span class="coding-sketch-node">choose B</span><span class="coding-sketch-node">choose C</span></div></div><p class="coding-sketch-note">add one choice, explore below it, then restore the parent path</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Path</span><strong>partial answer</strong><small>The current path is a valid unfinished choice.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Choose</span><strong>one branch</strong><small>Add one available value, cell, or letter.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Recurse</span><strong>smaller problem</strong><small>Explore everything below that choice.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Undo</span><strong>restore state</strong><small>Remove the same choice before the next branch.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>At every call, the path contains exactly the choices on the route from the root.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The visual is a choice tree, not a list of magic loops. Backtracking works because every branch starts from the same restored parent state. For this problem, hold onto the concrete trace: trace C-A-T through neighboring cells, marking each chosen cell temporarily.</figcaption></figure>

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
