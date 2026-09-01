---
title: "Implement Trie"
description: "Support word insert, full-word search, and prefix search."
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

> Support word insert, full-word search, and prefix search.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:implement-trie-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="implement-trie-state-title"><p class="visual-kicker">Prefixes share a path</p><p class="visual-title" id="implement-trie-state-title">Implement Trie: Store each character once along a shared prefix route</p><div class="coding-visual coding-visual--trie" data-coding-visual data-coding-mode="trie" data-coding-slug="implement-trie" role="group" aria-label="Implement Trie: insert cat and car -&gt; c-a is shared, then branch at the final letter. The path from the root to the current node spells exactly the prefix being queried."><div class="coding-visual-example"><span>Concrete trace</span><strong>insert cat and car -&gt; c-a is shared, then branch at the final letter</strong></div><div class="coding-visual-sketch coding-visual-sketch--trie"><div class="coding-sketch-prefix"><span class="coding-sketch-node">root</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node coding-sketch-node--state">c-a</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node coding-sketch-node--active">t / r</span></div><p class="coding-sketch-note">shared prefixes stay shared until the words branch</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Root</span><strong>empty prefix</strong><small>All words begin at one shared node.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Walk</span><strong>one character</strong><small>Follow the edge for the next letter.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Branch</span><strong>shared or new</strong><small>Reuse a prefix or create a child node.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Mark</span><strong>word ends here</strong><small>Separate a complete word from its prefix.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The path from the root to the current node spells exactly the prefix being queried.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> A trie turns repeated string prefixes into shared structure. A full-word marker matters because a word can end before another word that continues through it. For this problem, hold onto the concrete trace: insert cat and car -&gt; c-a is shared, then branch at the final letter.</figcaption></figure>

**Pattern:** Tree of character maps.

**Simple idea:** Follow one child map per character. Add an end marker after the last
character so a full word can be different from its prefix.

```python
class Trie:
   def __init__(self) -> None:
      self.root: dict = {}

   def insert(self, word: str) -> None:
      node = self.root
      for char in word:
         node = node.setdefault(char, {})
      node[None] = True

   def _walk(self, text: str) -> dict | None:
      node = self.root
      for char in text:
         if char not in node:
            return None
         node = node[char]
      return node

   def search(self, word: str) -> bool:
      node = self._walk(word)
      return node is not None and None in node

   def starts_with(self, prefix: str) -> bool:
      return self._walk(prefix) is not None
```

**Cost:** $O(L)$ time for each operation and $O(total characters)$ stored space.
