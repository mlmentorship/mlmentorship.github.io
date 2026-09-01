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
<figure class="learning-figure coding-visual-figure" aria-labelledby="graph-valid-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="graph-valid-tree-state-title">Graph Valid Tree: A valid tree needs exactly n-1 edges and one connected component.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="graph-valid-tree" role="group" aria-label="Graph Valid Tree: A valid tree needs exactly n-1 edges and one connected component."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether undirected edges form one valid tree.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Check the edge count"><div class="coding-trace-frame-heading"><span>Check the edge count</span><strong>Five nodes require exactly four edges.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">0</span><span class="coding-trace-node">1</span><span class="coding-trace-node">2</span><span class="coding-trace-node">3</span><span class="coding-trace-node">4</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">0-1</span><span class="coding-trace-edge">0-2</span><span class="coding-trace-edge">0-3</span><span class="coding-trace-edge">1-4</span></div></div><div class="coding-trace-meta"><span><b>detail</b>edges = 4 = n-1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Traverse once"><div class="coding-trace-frame-heading"><span>Traverse once</span><strong>DFS from 0 reaches every node without finding a cycle.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node is-state">0</span><span class="coding-trace-node is-state">1</span><span class="coding-trace-node is-state">2</span><span class="coding-trace-node is-state">3</span><span class="coding-trace-node is-state">4</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">0-1</span><span class="coding-trace-edge">0-2</span><span class="coding-trace-edge">0-3</span><span class="coding-trace-edge">1-4</span></div><div class="coding-trace-meta"><span><b>visited</b>0, 1, 2, 3, 4</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Accept the tree"><div class="coding-trace-frame-heading"><span>Accept the tree</span><strong>Correct edge count plus full reachability proves a tree.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">0</span><span class="coding-trace-node">1</span><span class="coding-trace-node">2</span><span class="coding-trace-node">3</span><span class="coding-trace-node">4</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">0-1</span><span class="coding-trace-edge">0-2</span><span class="coding-trace-edge">0-3</span><span class="coding-trace-edge">1-4</span></div></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Check the edge count</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Traverse once</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Accept the tree</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A valid tree needs exactly n-1 edges and one connected component.</p></div><figcaption><strong>Read it this way:</strong> Five nodes require exactly four edges. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
