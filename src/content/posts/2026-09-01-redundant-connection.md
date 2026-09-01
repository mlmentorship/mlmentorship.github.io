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
<figure class="learning-figure coding-visual-figure" aria-labelledby="redundant-connection-state-title"><p class="visual-kicker">Components with one representative</p><p class="visual-title" id="redundant-connection-state-title">Redundant Connection: Ask roots whether two endpoints already belong together</p><div class="coding-visual coding-visual--union" data-coding-visual data-coding-mode="union" data-coding-slug="redundant-connection" role="group" aria-label="Redundant Connection: edge 2-3 finds the same root at both ends -&gt; it closes the cycle. All nodes in one connected component eventually point to the same root."><div class="coding-visual-example"><span>Concrete trace</span><strong>edge 2-3 finds the same root at both ends -&gt; it closes the cycle</strong></div><div class="coding-visual-sketch coding-visual-sketch--union"><div class="coding-sketch-components"><span class="coding-sketch-component"><b>root A</b> · a · b</span><span class="coding-sketch-component"><b>root B</b> · c</span><span class="coding-sketch-component coding-sketch-component--active"><b>same root?</b> cycle</span></div><p class="coding-sketch-note">compare representatives before joining two components</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Find</span><strong>root(a), root(b)</strong><small>Follow parent links to each component representative.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Compare</span><strong>same root?</strong><small>Equal roots mean the edge closes a cycle.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Join</span><strong>different roots</strong><small>Attach one component under the other.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Compress</span><strong>short parent paths</strong><small>Future root checks become cheaper.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>All nodes in one connected component eventually point to the same root.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The parent array is a map of component identity. You do not need to walk every edge again; compare representatives and merge only when the groups differ. For this problem, hold onto the concrete trace: edge 2-3 finds the same root at both ends -&gt; it closes the cycle.</figcaption></figure>

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
