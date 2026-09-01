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
<figure class="learning-figure coding-visual-figure" aria-labelledby="design-add-and-search-words-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="design-add-and-search-words-state-title">Design Add and Search Words: A literal follows one trie child; a dot branches over every child.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="design-add-and-search-words" role="group" aria-label="Design Add and Search Words: A literal follows one trie child; a dot branches over every child."><div class="coding-visual-example"><span>Input and goal</span><strong>Store words and support `.` as a wildcard that matches any one character.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Store words"><div class="coding-trace-frame-heading"><span>Store words</span><strong>bad, dad, and mad share the suffix ad after different first letters.</strong></div><div class="coding-trace-trie"><div class="coding-trace-trie-path"><span class="coding-trace-trie-word">bad</span><span class="coding-trace-link-arrow">&rarr;</span><strong>b-a-d</strong></div><div class="coding-trace-trie-path"><span class="coding-trace-trie-word">dad</span><span class="coding-trace-link-arrow">&rarr;</span><strong>d-a-d</strong></div><div class="coding-trace-trie-path"><span class="coding-trace-trie-word">mad</span><span class="coding-trace-link-arrow">&rarr;</span><strong>m-a-d</strong></div></div><div class="coding-trace-meta"><span><b>action</b>insert three words</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Match a wildcard"><div class="coding-trace-frame-heading"><span>Match a wildcard</span><strong>For .ad, the dot tries b, d, and m, then follows a-d.</strong></div><div class="coding-trace-trie"><div class="coding-trace-trie-path"><span class="coding-trace-trie-word trace-tone-focus">.ad</span><span class="coding-trace-link-arrow">&rarr;</span><strong>b/d/m -&gt; a -&gt; d</strong></div></div><div class="coding-trace-meta"><span><b>query</b>.ad</span><span><b>action</b>branch at dot</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return true"><div class="coding-trace-frame-heading"><span>Return true</span><strong>One wildcard branch reaches a terminal word marker.</strong></div><div class="coding-trace-trie"><div class="coding-trace-trie-path"><span class="coding-trace-trie-word trace-tone-output">bad</span><span class="coding-trace-link-arrow">&rarr;</span><strong>b-a-d</strong></div><div class="coding-trace-trie-path"><span class="coding-trace-trie-word">dad</span><span class="coding-trace-link-arrow">&rarr;</span><strong>d-a-d</strong></div><div class="coding-trace-trie-path"><span class="coding-trace-trie-word">mad</span><span class="coding-trace-link-arrow">&rarr;</span><strong>m-a-d</strong></div></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Store words</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Match a wildcard</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return true</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A literal follows one trie child; a dot branches over every child.</p></div><figcaption><strong>Read it this way:</strong> bad, dad, and mad share the suffix ad after different first letters. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
