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
<figure class="learning-figure coding-visual-figure" aria-labelledby="implement-trie-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="implement-trie-state-title">Implement Trie: A shared character path stores prefixes once, with a terminal marker for complete words.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="implement-trie" role="group" aria-label="Implement Trie: A shared character path stores prefixes once, with a terminal marker for complete words."><div class="coding-visual-example"><span>Input and goal</span><strong>Support word insert, full-word search, and prefix search.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Insert cat"><div class="coding-trace-frame-heading"><span>Insert cat</span><strong>The path root-c-a-t is created and t gets an end marker.</strong></div><div class="coding-trace-trie"><div class="coding-trace-trie-path"><span class="coding-trace-trie-word trace-tone-focus">cat</span><span class="coding-trace-link-arrow">&rarr;</span><strong>c-a-t</strong></div></div><div class="coding-trace-meta"><span><b>action</b>insert cat</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Share c-a"><div class="coding-trace-frame-heading"><span>Share c-a</span><strong>Inserting car reuses c-a and branches only at the final character.</strong></div><div class="coding-trace-trie"><div class="coding-trace-trie-path"><span class="coding-trace-trie-word trace-tone-state">cat</span><span class="coding-trace-link-arrow">&rarr;</span><strong>c-a-t</strong></div><div class="coding-trace-trie-path"><span class="coding-trace-trie-word trace-tone-focus">car</span><span class="coding-trace-link-arrow">&rarr;</span><strong>c-a-r</strong></div></div><div class="coding-trace-meta"><span><b>action</b>share prefix c-a</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Search a prefix"><div class="coding-trace-frame-heading"><span>Search a prefix</span><strong>starts_with(&quot;ca&quot;) succeeds even before choosing t or r.</strong></div><div class="coding-trace-trie"><div class="coding-trace-trie-path"><span class="coding-trace-trie-word trace-tone-output">cat</span><span class="coding-trace-link-arrow">&rarr;</span><strong>c-a-t</strong></div><div class="coding-trace-trie-path"><span class="coding-trace-trie-word trace-tone-output">car</span><span class="coding-trace-link-arrow">&rarr;</span><strong>c-a-r</strong></div></div><div class="coding-trace-meta"><span><b>query</b>ca</span><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Insert cat</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Share c-a</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Search a prefix</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A shared character path stores prefixes once, with a terminal marker for complete words.</p></div><figcaption><strong>Read it this way:</strong> The path root-c-a-t is created and t gets an end marker. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
