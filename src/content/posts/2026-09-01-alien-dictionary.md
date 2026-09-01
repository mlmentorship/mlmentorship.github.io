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
<figure class="learning-figure coding-visual-figure" aria-labelledby="alien-dictionary-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="alien-dictionary-state-title">Alien Dictionary: The first differing character in adjacent words creates a directed ordering edge.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="alien-dictionary" role="group" aria-label="Alien Dictionary: The first differing character in adjacent words creates a directed ordering edge."><div class="coding-visual-example"><span>Input and goal</span><strong>Infer character order from words that are sorted in an unknown alphabet.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Extract a rule"><div class="coding-trace-frame-heading"><span>Extract a rule</span><strong>wrt and wrf first differ at t and f, so t -&gt; f.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">w</span><span class="coding-trace-node">r</span><span class="coding-trace-node">t</span><span class="coding-trace-node">f</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">t -&gt; f</span></div></div><div class="coding-trace-meta"><span><b>rule</b>t before f</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Collect rules"><div class="coding-trace-frame-heading"><span>Collect rules</span><strong>The other adjacent differences add w-&gt;e, e-&gt;r, and r-&gt;t.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">w</span><span class="coding-trace-node">e</span><span class="coding-trace-node">r</span><span class="coding-trace-node">t</span><span class="coding-trace-node">f</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">w -&gt; e</span><span class="coding-trace-edge">e -&gt; r</span><span class="coding-trace-edge">r -&gt; t</span><span class="coding-trace-edge">t -&gt; f</span></div><div class="coding-trace-meta"><span><b>indegree</b>w:0, e:1, r:1, t:1, f:1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Topologically order"><div class="coding-trace-frame-heading"><span>Topologically order</span><strong>Remove zero-indegree letters and return a valid alien alphabet.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">w</span><span class="coding-trace-node">e</span><span class="coding-trace-node">r</span><span class="coding-trace-node">t</span><span class="coding-trace-node">f</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">w -&gt; e</span><span class="coding-trace-edge">e -&gt; r</span><span class="coding-trace-edge">r -&gt; t</span><span class="coding-trace-edge">t -&gt; f</span></div><div class="coding-trace-meta"><span><b>order</b>w, e, r, t, f</span></div></div><div class="coding-trace-meta"><span><b>result</b>wertf</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Extract a rule</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Collect rules</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Topologically order</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>The first differing character in adjacent words creates a directed ordering edge.</p></div><figcaption><strong>Read it this way:</strong> wrt and wrf first differ at t and f, so t -&gt; f. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
