---
title: "Graph Valid Tree"
description: "Check whether undirected edges form one valid tree."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Mixed"
priority: "Core"
aliases: []
prerequisites: []
---

> Check whether undirected edges form one valid tree.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:graph-valid-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="graph-valid-tree-state-title"><p class="visual-kicker">Reachability without repetition</p><p class="visual-title" id="graph-valid-tree-state-title">Graph Valid Tree: Turn a large graph into one frontier and one visited set</p><div class="coding-visual coding-visual--graph" data-coding-visual data-coding-mode="graph" data-coding-slug="graph-valid-tree" role="group" aria-label="Graph Valid Tree: n nodes need n-1 edges; then one DFS must reach every node. Every visited node has been scheduled exactly once, so cycles cannot repeat work."><div class="coding-visual-example"><span>Concrete trace</span><strong>n nodes need n-1 edges; then one DFS must reach every node</strong></div><div class="coding-visual-sketch coding-visual-sketch--graph"><div class="coding-sketch-graph"><span class="coding-sketch-node coding-sketch-node--active">start</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node">visited</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node coding-sketch-node--state">unseen</span></div><p class="coding-sketch-note">the frontier separates visited nodes from reachable unknowns</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Start</span><strong>current node</strong><small>Choose a source or an unseen component.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expand</span><strong>neighbors</strong><small>Follow edges or legal grid moves.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Mark</span><strong>visited</strong><small>Record a node before adding it again.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Count</span><strong>component or goal</strong><small>The explored set gives the answer.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every visited node has been scheduled exactly once, so cycles cannot repeat work.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The frontier is the boundary between known and unknown nodes. Marking a node when it enters the frontier prevents a cycle from creating duplicate searches. For this problem, hold onto the concrete trace: n nodes need n-1 edges; then one DFS must reach every node.</figcaption></figure>

**Pattern:** Edge count plus DFS.

**Simple idea:** A tree with `n` nodes must have exactly `n - 1` edges. With that edge count,
the graph is a tree if DFS can reach every node.

```python
def valid_tree(node_count: int, edges: list[list[int]]) -> bool:
   if len(edges) != node_count - 1:
      return False

   graph = [[] for _ in range(node_count)]
   for first, second in edges:
      graph[first].append(second)
      graph[second].append(first)

   seen = {0}
   stack = [0]
   while stack:
      for neighbor in graph[stack.pop()]:
         if neighbor not in seen:
            seen.add(neighbor)
            stack.append(neighbor)
   return len(seen) == node_count
```

**Cost:** Close to $O(V + E)$ time and $O(V)$ space.
