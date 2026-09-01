---
title: "Clone Graph"
description: "Make a deep copy of a connected graph."
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

> Make a deep copy of a connected graph.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:clone-graph-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="clone-graph-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="clone-graph-state-title">Clone Graph: Map each original node to one copy, then connect copies using the map.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="clone-graph" role="group" aria-label="Clone Graph: Map each original node to one copy, then connect copies using the map."><div class="coding-visual-example"><span>Input and goal</span><strong>Make a deep copy of a connected graph.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Copy the start"><div class="coding-trace-frame-heading"><span>Copy the start</span><strong>Original node 1 gets exactly one copy before neighbors are explored.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node is-state">original 1</span><span class="coding-trace-node">original 2</span><span class="coding-trace-node">copy 1</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">1 &lt;-&gt; 2</span></div><div class="coding-trace-meta"><span><b>visited</b>original 1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Copy neighbors once"><div class="coding-trace-frame-heading"><span>Copy neighbors once</span><strong>When node 2 appears, create copy 2 and reuse copy 1 for the reverse edge.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">original 1</span><span class="coding-trace-node">original 2</span><span class="coding-trace-node">copy 1</span><span class="coding-trace-node">copy 2</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">copy 1 &lt;-&gt; copy 2</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return the copied component"><div class="coding-trace-frame-heading"><span>Return the copied component</span><strong>Every original edge has a matching copied edge.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">copy 1</span><span class="coding-trace-node">copy 2</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">copy 1 &lt;-&gt; copy 2</span></div></div><div class="coding-trace-meta"><span><b>result</b>deep copy</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Copy the start</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Copy neighbors once</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the copied component</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Map each original node to one copy, then connect copies using the map.</p></div><figcaption><strong>Read it this way:</strong> Original node 1 gets exactly one copy before neighbors are explored. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Graph traversal plus a map from old nodes to copied nodes.

**Simple idea:** The map has two jobs. It prevents repeated work and gives the copy of every
neighbor when building copied edges.

```python
from __future__ import annotations
from collections import deque

class GraphNode:
   def __init__(self, val: int = 0, neighbors: list[GraphNode] | None = None) -> None:
      self.val = val
      self.neighbors = neighbors or []


def clone_graph(node: GraphNode | None) -> GraphNode | None:
   if node is None:
      return None

   copies = {node: GraphNode(node.val)}
   queue = deque([node])

   while queue:
      current = queue.popleft()
      for neighbor in current.neighbors:
         if neighbor not in copies:
            copies[neighbor] = GraphNode(neighbor.val)
            queue.append(neighbor)
         copies[current].neighbors.append(copies[neighbor])
   return copies[node]
```

**Cost:** $O(V + E)$ time and $O(V)$ space.
