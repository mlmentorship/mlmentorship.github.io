---
title: "Design Add and Search Words"
description: "Store words and support `.` as a wildcard that matches any one character."
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

> Store words and support `.` as a wildcard that matches any one character.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:design-add-and-search-words-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="design-add-and-search-words-state-title"><p class="visual-kicker">Prefixes share a path</p><p class="visual-title" id="design-add-and-search-words-state-title">Design Add and Search Words: Store each character once along a shared prefix route</p><div class="coding-visual coding-visual--trie" data-coding-visual data-coding-mode="trie" data-coding-slug="design-add-and-search-words" role="group" aria-label="Design Add and Search Words: search c.t -&gt; dot branches across every child at position 2. The path from the root to the current node spells exactly the prefix being queried."><div class="coding-visual-example"><span>Concrete trace</span><strong>search c.t -&gt; dot branches across every child at position 2</strong></div><div class="coding-visual-sketch coding-visual-sketch--trie"><div class="coding-sketch-prefix"><span class="coding-sketch-node">root</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node coding-sketch-node--state">c-a</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node coding-sketch-node--active">t / r</span></div><p class="coding-sketch-note">shared prefixes stay shared until the words branch</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Root</span><strong>empty prefix</strong><small>All words begin at one shared node.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Walk</span><strong>one character</strong><small>Follow the edge for the next letter.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Branch</span><strong>shared or new</strong><small>Reuse a prefix or create a child node.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Mark</span><strong>word ends here</strong><small>Separate a complete word from its prefix.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The path from the root to the current node spells exactly the prefix being queried.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> A trie turns repeated string prefixes into shared structure. A full-word marker matters because a word can end before another word that continues through it. For this problem, hold onto the concrete trace: search c.t -&gt; dot branches across every child at position 2.</figcaption></figure>

**Pattern:** Trie plus DFS when a wildcard appears.

**Simple idea:** Normal letters follow one child. A dot tries every child. The end marker
still checks that the full word length matched.

```python
class WordDictionary:
   def __init__(self) -> None:
      self.root: dict = {}

   def add_word(self, word: str) -> None:
      node = self.root
      for char in word:
         node = node.setdefault(char, {})
      node[None] = True

   def search(self, word: str) -> bool:
      def match(index: int, node: dict) -> bool:
         if index == len(word):
            return None in node
         if word[index] == ".":
            return any(match(index + 1, child) for key, child in node.items() if key)
         return word[index] in node and match(index + 1, node[word[index]])

      return match(0, self.root)
```

**Cost:** Adding takes $O(L)$ time. A normal search takes $O(L)$. Many wildcards can make
search exponential in the word length.
