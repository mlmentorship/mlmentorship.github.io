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
<figure class="learning-figure coding-visual-figure" aria-labelledby="word-search-ii-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="word-search-ii-state-title">Word Search II: A trie shares word prefixes and stops a board search as soon as the path leaves the trie.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="word-search-ii" role="group" tabindex="0" aria-label="Word Search II: A trie shares word prefixes and stops a board search as soon as the path leaves the trie."><div class="coding-visual-example"><span>Input and goal</span><strong>Find every dictionary word that can be formed on a letter board.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Build shared search structure"><div class="coding-trace-frame-heading"><span>Build shared search structure</span><strong>All dictionary words enter one trie; each board path follows only a matching child.</strong></div><div class="coding-trace-trie"><svg viewBox="0 0 560 116" role="img" aria-label="Trie prefix topology"><line class="coding-trace-edge-line" x1="55" y1="30" x2="110" y2="30" /><line class="coding-trace-edge-line" x1="110" y1="30" x2="165" y2="30" /><line class="coding-trace-edge-line" x1="165" y1="30" x2="220" y2="30" /><g data-motion-key="trie-oath-0"><circle cx="55" cy="30" r="16" /><text x="55" y="34">o</text></g><g data-motion-key="trie-oath-1"><circle cx="110" cy="30" r="16" /><text x="110" y="34">a</text></g><g data-motion-key="trie-oath-2"><circle cx="165" cy="30" r="16" /><text x="165" y="34">t</text></g><g data-motion-key="trie-oath-3"><circle cx="220" cy="30" r="16" /><text x="220" y="34">h</text></g><text class="coding-trace-node-state" x="300" y="34">o-a-t-h</text><line class="coding-trace-edge-line" x1="55" y1="88" x2="110" y2="88" /><line class="coding-trace-edge-line" x1="110" y1="88" x2="165" y2="88" /><g data-motion-key="trie-eat-0"><circle cx="55" cy="88" r="16" /><text x="55" y="92">e</text></g><g data-motion-key="trie-eat-1"><circle cx="110" cy="88" r="16" /><text x="110" y="92">a</text></g><g data-motion-key="trie-eat-2"><circle cx="165" cy="88" r="16" /><text x="165" y="92">t</text></g><text class="coding-trace-node-state" x="245" y="92">e-a-t</text></svg></div><div class="coding-trace-meta"><span><b>action</b>trie prefixes</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Walk the board and trie together"><div class="coding-trace-frame-heading"><span>Walk the board and trie together</span><strong>A board path that reaches o-a-t may continue to h; a path with no trie child stops.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="marker-o"><span>o</span><small>o</small></span><span class="coding-trace-grid-cell trace-tone-state" data-motion-key="marker-a"><span>a</span><small>a</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>a</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>n</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>e</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="marker-t"><span>t</span><small>t</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>a</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>e</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>i</span></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="marker-h"><span>h</span><small>h</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>k</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>r</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-0"><span>i</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>f</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-2"><span>l</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-3"><span>v</span></span></div><div class="coding-trace-meta"><span><b>path</b>oath</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Emit each terminal word once"><div class="coding-trace-frame-heading"><span>Emit each terminal word once</span><strong>The board finds oath and eat; failed prefixes never expand further.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>o</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>a</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>a</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>n</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>e</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>t</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>a</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>e</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>i</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>h</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>k</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>r</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-0"><span>i</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-1"><span>f</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-2"><span>l</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-3-3"><span>v</span></span></div><div class="coding-trace-meta"><span><b>result</b>[oath,eat]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Build shared search structure</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Walk the board and trie together</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Emit each terminal word once</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A trie shares word prefixes and stops a board search as soon as the path leaves the trie.</p></div><figcaption><strong>Read it this way:</strong> All dictionary words enter one trie; each board path follows only a matching child. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
