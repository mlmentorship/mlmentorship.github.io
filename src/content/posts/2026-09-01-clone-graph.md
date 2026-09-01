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
<figure class="learning-figure coding-visual-figure" aria-labelledby="clone-graph-state-title"><p class="visual-kicker">Reachability without repetition</p><p class="visual-title" id="clone-graph-state-title">Clone Graph: Turn a large graph into one frontier and one visited set</p><div class="coding-visual coding-visual--graph" data-coding-visual data-coding-mode="graph" data-coding-slug="clone-graph" role="group" aria-label="Clone Graph: copy node 1 once, then point its copy at copies of every neighbor. Every visited node has been scheduled exactly once, so cycles cannot repeat work."><div class="coding-visual-example"><span>Concrete trace</span><strong>copy node 1 once, then point its copy at copies of every neighbor</strong></div><div class="coding-visual-sketch coding-visual-sketch--graph"><div class="coding-sketch-graph"><span class="coding-sketch-node coding-sketch-node--active">start</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node">visited</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node coding-sketch-node--state">unseen</span></div><p class="coding-sketch-note">the frontier separates visited nodes from reachable unknowns</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Start</span><strong>current node</strong><small>Choose a source or an unseen component.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expand</span><strong>neighbors</strong><small>Follow edges or legal grid moves.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Mark</span><strong>visited</strong><small>Record a node before adding it again.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Count</span><strong>component or goal</strong><small>The explored set gives the answer.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every visited node has been scheduled exactly once, so cycles cannot repeat work.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The frontier is the boundary between known and unknown nodes. Marking a node when it enters the frontier prevents a cycle from creating duplicate searches. For this problem, hold onto the concrete trace: copy node 1 once, then point its copy at copies of every neighbor.</figcaption></figure>

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
