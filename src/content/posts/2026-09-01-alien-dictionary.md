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
<figure class="learning-figure coding-visual-figure" aria-labelledby="alien-dictionary-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="alien-dictionary-state-title">Alien Dictionary: The first differing character in adjacent words creates a directed ordering edge.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="alien-dictionary" role="group" tabindex="0" aria-label="Alien Dictionary: The first differing character in adjacent words creates a directed ordering edge."><div class="coding-visual-example"><span>Input and goal</span><strong>Infer character order from words that are sorted in an unknown alphabet.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Extract a rule"><div class="coding-trace-frame-heading"><span>Extract a rule</span><strong>wrt and wrf first differ at t and f, so t -&gt; f.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-t -&gt; f-0"><line class="coding-trace-edge-line" x1="240" y1="193" x2="70" y2="115.00000000000001" /><text x="155" y="148">t -&gt; f</text></g><g class="coding-trace-graph-node" data-motion-key="node-w-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">w</text></g><g class="coding-trace-graph-node" data-motion-key="node-r-1"><circle cx="410" cy="115" r="23" /><text x="410" y="119">r</text></g><g class="coding-trace-graph-node" data-motion-key="node-t-2"><circle cx="240" cy="193" r="23" /><text x="240" y="197">t</text></g><g class="coding-trace-graph-node" data-motion-key="node-f-3"><circle cx="70" cy="115.00000000000001" r="23" /><text x="70" y="119.00000000000001">f</text></g></svg></div><div class="coding-trace-meta"><span><b>rule</b>t before f</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Collect rules"><div class="coding-trace-frame-heading"><span>Collect rules</span><strong>The other adjacent differences add w-&gt;e, e-&gt;r, and r-&gt;t.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-w -&gt; e-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="401.67960777017606" y2="90.8966744387541" /><text x="320.83980388508803" y="57.94833721937705">w -&gt; e</text></g><g data-motion-key="edge-e -&gt; r-1"><line class="coding-trace-edge-line" x1="401.67960777017606" y1="90.8966744387541" x2="339.9234928897204" y2="178.1033255612459" /><text x="370.8015503299482" y="128.5">e -&gt; r</text></g><g data-motion-key="edge-r -&gt; t-2"><line class="coding-trace-edge-line" x1="339.9234928897204" y1="178.1033255612459" x2="140.07650711027958" y2="178.1033255612459" /><text x="240" y="172.1033255612459">r -&gt; t</text></g><g data-motion-key="edge-t -&gt; f-3"><line class="coding-trace-edge-line" x1="140.07650711027958" y1="178.1033255612459" x2="78.32039222982388" y2="90.8966744387541" /><text x="109.19844967005173" y="128.5">t -&gt; f</text></g><g class="coding-trace-graph-node" data-motion-key="node-w-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">w</text></g><g class="coding-trace-graph-node" data-motion-key="node-e-1"><circle cx="401.67960777017606" cy="90.8966744387541" r="23" /><text x="401.67960777017606" y="94.8966744387541">e</text></g><g class="coding-trace-graph-node" data-motion-key="node-r-2"><circle cx="339.9234928897204" cy="178.1033255612459" r="23" /><text x="339.9234928897204" y="182.1033255612459">r</text></g><g class="coding-trace-graph-node" data-motion-key="node-t-3"><circle cx="140.07650711027958" cy="178.1033255612459" r="23" /><text x="140.07650711027958" y="182.1033255612459">t</text></g><g class="coding-trace-graph-node" data-motion-key="node-f-4"><circle cx="78.32039222982388" cy="90.8966744387541" r="23" /><text x="78.32039222982388" y="94.8966744387541">f</text></g></svg><div class="coding-trace-meta"><span><b>indegree</b>w:0, e:1, r:1, t:1, f:1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Topologically order"><div class="coding-trace-frame-heading"><span>Topologically order</span><strong>Remove zero-indegree letters and return a valid alien alphabet.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-w -&gt; e-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="401.67960777017606" y2="90.8966744387541" /><text x="320.83980388508803" y="57.94833721937705">w -&gt; e</text></g><g data-motion-key="edge-e -&gt; r-1"><line class="coding-trace-edge-line" x1="401.67960777017606" y1="90.8966744387541" x2="339.9234928897204" y2="178.1033255612459" /><text x="370.8015503299482" y="128.5">e -&gt; r</text></g><g data-motion-key="edge-r -&gt; t-2"><line class="coding-trace-edge-line" x1="339.9234928897204" y1="178.1033255612459" x2="140.07650711027958" y2="178.1033255612459" /><text x="240" y="172.1033255612459">r -&gt; t</text></g><g data-motion-key="edge-t -&gt; f-3"><line class="coding-trace-edge-line" x1="140.07650711027958" y1="178.1033255612459" x2="78.32039222982388" y2="90.8966744387541" /><text x="109.19844967005173" y="128.5">t -&gt; f</text></g><g class="coding-trace-graph-node" data-motion-key="node-w-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">w</text></g><g class="coding-trace-graph-node" data-motion-key="node-e-1"><circle cx="401.67960777017606" cy="90.8966744387541" r="23" /><text x="401.67960777017606" y="94.8966744387541">e</text></g><g class="coding-trace-graph-node" data-motion-key="node-r-2"><circle cx="339.9234928897204" cy="178.1033255612459" r="23" /><text x="339.9234928897204" y="182.1033255612459">r</text></g><g class="coding-trace-graph-node" data-motion-key="node-t-3"><circle cx="140.07650711027958" cy="178.1033255612459" r="23" /><text x="140.07650711027958" y="182.1033255612459">t</text></g><g class="coding-trace-graph-node" data-motion-key="node-f-4"><circle cx="78.32039222982388" cy="90.8966744387541" r="23" /><text x="78.32039222982388" y="94.8966744387541">f</text></g></svg><div class="coding-trace-meta"><span><b>order</b>w, e, r, t, f</span></div></div><div class="coding-trace-meta"><span><b>result</b>wertf</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Extract a rule</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Collect rules</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Topologically order</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>The first differing character in adjacent words creates a directed ordering edge.</p></div><figcaption><strong>Read it this way:</strong> wrt and wrf first differ at t and f, so t -&gt; f. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
