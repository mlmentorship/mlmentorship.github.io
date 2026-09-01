---
title: "Alien Dictionary"
description: "Infer character order from words that are sorted in an unknown alphabet."
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

> Infer character order from words that are sorted in an unknown alphabet.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:alien-dictionary-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="alien-dictionary-state-title"><p class="visual-kicker">Dependencies becoming ready</p><p class="visual-title" id="alien-dictionary-state-title">Alien Dictionary: Remove prerequisites until the next zero-indegree item appears</p><div class="coding-visual coding-visual--topology" data-coding-visual data-coding-mode="topology" data-coding-slug="alien-dictionary" role="group" aria-label="Alien Dictionary: first difference in w-r and e-r gives w&lt;e; topological sort orders the alphabet. The ready queue contains exactly the nodes whose prerequisites are complete."><div class="coding-visual-example"><span>Concrete trace</span><strong>first difference in w-r and e-r gives w&lt;e; topological sort orders the alphabet</strong></div><div class="coding-visual-sketch coding-visual-sketch--topology"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">0 unmet</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--active">ready</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill">next is ready</span></div><p class="coding-sketch-note">remove incoming requirements until a node becomes ready</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Count</span><strong>incoming edges</strong><small>Record how many requirements each node still has.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Ready</span><strong>indegree = 0</strong><small>Only nodes with no unmet prerequisite can start.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Remove</span><strong>complete one</strong><small>Subtract its edge from every dependent node.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Detect</span><strong>cycle or order</strong><small>Unfinished nodes reveal a dependency cycle.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The ready queue contains exactly the nodes whose prerequisites are complete.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Imagine removing foundation blocks from a dependency wall. A block becomes available only when every incoming requirement has disappeared. For this problem, hold onto the concrete trace: first difference in w-r and e-r gives w&lt;e; topological sort orders the alphabet.</figcaption></figure>

**Pattern:** Build character rules, then topological sort.

**Simple idea:** Compare each neighboring pair of words. Their first different characters
give one order rule. Include characters with no edges. Reject a longer word placed
before
its exact prefix.

```python
from collections import deque

def _first_difference(first: str, second: str) -> tuple[str, str] | None:
   for first_char, second_char in zip(first, second, strict=False):
      if first_char != second_char:
         return first_char, second_char
   return None


def alien_order(words: list[str]) -> str:
   graph = {char: set() for word in words for char in word}
   indegree = {char: 0 for char in graph}

   for first, second in zip(words, words[1:], strict=False):
      difference = _first_difference(first, second)
      if difference is None:
         if len(first) > len(second):
            return ""
         continue

      first_char, second_char = difference
      if second_char not in graph[first_char]:
         graph[first_char].add(second_char)
         indegree[second_char] += 1

   ready = deque(char for char, count in indegree.items() if count == 0)
   order: list[str] = []

   while ready:
      char = ready.popleft()
      order.append(char)
      for next_char in graph[char]:
         indegree[next_char] -= 1
         if indegree[next_char] == 0:
            ready.append(next_char)

   return "".join(order) if len(order) == len(indegree) else ""
```

**Cost:** $O(C + E)$ time and space, where $C$ is the total number of characters read.
