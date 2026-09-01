---
title: "Redundant Connection"
description: "Find the edge that creates a cycle in an undirected graph."
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

> Find the edge that creates a cycle in an undirected graph.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:redundant-connection-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="redundant-connection-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="redundant-connection-state-title">Redundant Connection: An edge is redundant when both endpoints already have the same representative root.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="redundant-connection" role="group" aria-label="Redundant Connection: An edge is redundant when both endpoints already have the same representative root."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the edge that creates a cycle in an undirected graph.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Join separate components"><div class="coding-trace-frame-heading"><span>Join separate components</span><strong>Edges 1-2 and 1-3 create one component rooted at 1.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">1</span><span class="coding-trace-node">2</span><span class="coding-trace-node">3</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">1 - 2</span><span class="coding-trace-edge">1 - 3</span></div><div class="coding-trace-meta"><span><b>components</b>root 1: {1,2,3}</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Test the closing edge"><div class="coding-trace-frame-heading"><span>Test the closing edge</span><strong>For edge 2-3, find(2) and find(3) both return root 1.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">1</span><span class="coding-trace-node">2</span><span class="coding-trace-node">3</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">1 - 2</span><span class="coding-trace-edge">1 - 3</span><span class="coding-trace-edge">2 - 3</span></div><div class="coding-trace-meta"><span><b>roots</b>2 -&gt; 1, 3 -&gt; 1</span></div></div><div class="coding-trace-meta"><span><b>current</b>2 - 3</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Reject the cycle edge"><div class="coding-trace-frame-heading"><span>Reject the cycle edge</span><strong>Adding 2-3 would close a cycle, so return it.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">1</span><span class="coding-trace-node">2</span><span class="coding-trace-node">3</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">1 - 2</span><span class="coding-trace-edge">1 - 3</span><span class="coding-trace-edge">2 - 3</span></div></div><div class="coding-trace-meta"><span><b>current</b>2 - 3</span><span><b>result</b>[2,3]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Join separate components</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Test the closing edge</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Reject the cycle edge</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>An edge is redundant when both endpoints already have the same representative root.</p></div><figcaption><strong>Read it this way:</strong> Edges 1-2 and 1-3 create one component rooted at 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Union-find.

**Simple idea:** Before adding an edge, check whether both ends already have the same root.
If they do, the edge closes a cycle.

```python
class DisjointSet:
   def __init__(self, size: int) -> None:
      self.parent = list(range(size))
      self.component_size = [1] * size

   def find(self, node: int) -> int:
      while node != self.parent[node]:
         self.parent[node] = self.parent[self.parent[node]]
         node = self.parent[node]
      return node

   def union(self, first: int, second: int) -> bool:
      first_root = self.find(first)
      second_root = self.find(second)
      if first_root == second_root:
         return False

      if self.component_size[first_root] < self.component_size[second_root]:
         first_root, second_root = second_root, first_root
      self.parent[second_root] = first_root
      self.component_size[first_root] += self.component_size[second_root]
      return True


def find_redundant_connection(edges: list[list[int]]) -> list[int]:
   if not edges:
      return []

   groups = DisjointSet(max(max(edge) for edge in edges) + 1)
   for first, second in edges:
      if not groups.union(first, second):
         return [first, second]
   return []
```

**Cost:** Close to $O(E)$ time and $O(V)$ space.
