---
title: "Number of Connected Components"
description: "Count separate groups in an undirected graph."
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

> Count separate groups in an undirected graph.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:number-of-connected-components-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="number-of-connected-components-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="number-of-connected-components-state-title">Number of Connected Components: Every unseen node starts one DFS component and marks its whole group.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="number-of-connected-components" role="group" aria-label="Number of Connected Components: Every unseen node starts one DFS component and marks its whole group."><div class="coding-visual-example"><span>Input and goal</span><strong>Count separate groups in an undirected graph.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Start component 1"><div class="coding-trace-frame-heading"><span>Start component 1</span><strong>Node 0 reaches 1 and 2.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node is-state">0</span><span class="coding-trace-node is-state">1</span><span class="coding-trace-node is-state">2</span><span class="coding-trace-node">3</span><span class="coding-trace-node">4</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">0-1</span><span class="coding-trace-edge">1-2</span><span class="coding-trace-edge">3-4</span></div><div class="coding-trace-meta"><span><b>visited</b>0, 1, 2</span><span><b>components</b>1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Find the next unseen node"><div class="coding-trace-frame-heading"><span>Find the next unseen node</span><strong>Node 3 starts a second flood and reaches 4.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node is-state">0</span><span class="coding-trace-node is-state">1</span><span class="coding-trace-node is-state">2</span><span class="coding-trace-node is-state">3</span><span class="coding-trace-node is-state">4</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">0-1</span><span class="coding-trace-edge">1-2</span><span class="coding-trace-edge">3-4</span></div><div class="coding-trace-meta"><span><b>visited</b>0, 1, 2, 3, 4</span><span><b>components</b>2</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return the count"><div class="coding-trace-frame-heading"><span>Return the count</span><strong>Two starting floods mean two connected components.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">0</span><span class="coding-trace-node">1</span><span class="coding-trace-node">2</span><span class="coding-trace-node">3</span><span class="coding-trace-node">4</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">0-1</span><span class="coding-trace-edge">1-2</span><span class="coding-trace-edge">3-4</span></div></div><div class="coding-trace-meta"><span><b>result</b>2</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start component 1</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Find the next unseen node</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the count</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Every unseen node starts one DFS component and marks its whole group.</p></div><figcaption><strong>Read it this way:</strong> Node 0 reaches 1 and 2. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** DFS from every unseen node.

**Simple idea:** Every unseen node starts one new component. DFS marks its full group, so no
node in that group starts another component.

```python
def count_components(node_count: int, edges: list[list[int]]) -> int:
   graph = [[] for _ in range(node_count)]
   for first, second in edges:
      graph[first].append(second)
      graph[second].append(first)

   seen: set[int] = set()
   components = 0
   for start in range(node_count):
      if start in seen:
         continue

      components += 1
      seen.add(start)
      stack = [start]
      while stack:
         for neighbor in graph[stack.pop()]:
            if neighbor not in seen:
               seen.add(neighbor)
               stack.append(neighbor)
   return components
```

**Cost:** Close to $O(V + E)$ time and $O(V)$ space.
