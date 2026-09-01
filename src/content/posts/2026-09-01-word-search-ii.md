---
title: "Word Search II"
description: "Find every dictionary word that can be formed on a letter board."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

> Find every dictionary word that can be formed on a letter board.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:word-search-ii-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="word-search-ii-state-title"><p class="visual-kicker">Prefixes share a path</p><p class="visual-title" id="word-search-ii-state-title">Word Search II: Store each character once along a shared prefix route</p><div class="coding-visual coding-visual--trie" data-coding-visual data-coding-mode="trie" data-coding-slug="word-search-ii" role="group" aria-label="Word Search II: the trie rejects a board path as soon as it is not a dictionary prefix. The path from the root to the current node spells exactly the prefix being queried."><div class="coding-visual-example"><span>Concrete trace</span><strong>the trie rejects a board path as soon as it is not a dictionary prefix</strong></div><div class="coding-visual-sketch coding-visual-sketch--trie"><div class="coding-sketch-prefix"><span class="coding-sketch-node">root</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node coding-sketch-node--state">c-a</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node coding-sketch-node--active">t / r</span></div><p class="coding-sketch-note">shared prefixes stay shared until the words branch</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Root</span><strong>empty prefix</strong><small>All words begin at one shared node.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Walk</span><strong>one character</strong><small>Follow the edge for the next letter.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Branch</span><strong>shared or new</strong><small>Reuse a prefix or create a child node.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Mark</span><strong>word ends here</strong><small>Separate a complete word from its prefix.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The path from the root to the current node spells exactly the prefix being queried.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> A trie turns repeated string prefixes into shared structure. A full-word marker matters because a word can end before another word that continues through it. For this problem, hold onto the concrete trace: the trie rejects a board path as soon as it is not a dictionary prefix.</figcaption></figure>

**Pattern:** Trie plus grid backtracking.

**Simple idea:** Word Search starts a separate search for one word. A trie lets all words
share the same search. Stop when the current board path is not a dictionary prefix.

```python
def _make_word_trie(words: list[str]) -> dict:
   trie: dict = {}
   for word in words:
      node = trie
      for char in word:
         node = node.setdefault(char, {})
      node[None] = word
   return trie


def _find_words_from(
   board: list[list[str]], row: int, col: int, node: dict, answer: list[str]
) -> None:
   char = board[row][col]
   if char not in node:
      return

   next_node = node[char]
   word = next_node.pop(None, None)
   if word:
      answer.append(word)

   board[row][col] = "#"
   for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      next_row = row + row_step
      next_col = col + col_step
      if 0 <= next_row < len(board) and 0 <= next_col < len(board[0]):
         _find_words_from(board, next_row, next_col, next_node, answer)
   board[row][col] = char

   if not next_node:
      node.pop(char)


def find_words(board: list[list[str]], words: list[str]) -> list[str]:
   trie = _make_word_trie(words)
   answer: list[str] = []

   for row in range(len(board)):
      for col in range(len(board[0])):
         _find_words_from(board, row, col, trie, answer)
   return answer
```

**Cost:** The trie takes $O(total word characters)$ space. Search time depends on the board
and shared prefixes. Prefix checks remove most impossible paths early.
