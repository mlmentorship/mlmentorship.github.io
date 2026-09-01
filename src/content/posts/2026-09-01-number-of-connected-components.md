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
<figure class="learning-figure coding-visual-figure" aria-labelledby="number-of-connected-components-state-title"><p class="visual-kicker">Reachability without repetition</p><p class="visual-title" id="number-of-connected-components-state-title">Number of Connected Components: Turn a large graph into one frontier and one visited set</p><div class="coding-visual coding-visual--graph" data-coding-visual data-coding-mode="graph" data-coding-slug="number-of-connected-components" role="group" aria-label="Number of Connected Components: each unseen node starts a flood and increments the component count. Every visited node has been scheduled exactly once, so cycles cannot repeat work."><div class="coding-visual-example"><span>Concrete trace</span><strong>each unseen node starts a flood and increments the component count</strong></div><div class="coding-visual-sketch coding-visual-sketch--graph"><div class="coding-sketch-graph"><span class="coding-sketch-node coding-sketch-node--active">start</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node">visited</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node coding-sketch-node--state">unseen</span></div><p class="coding-sketch-note">the frontier separates visited nodes from reachable unknowns</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Start</span><strong>current node</strong><small>Choose a source or an unseen component.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expand</span><strong>neighbors</strong><small>Follow edges or legal grid moves.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Mark</span><strong>visited</strong><small>Record a node before adding it again.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Count</span><strong>component or goal</strong><small>The explored set gives the answer.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every visited node has been scheduled exactly once, so cycles cannot repeat work.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The frontier is the boundary between known and unknown nodes. Marking a node when it enters the frontier prevents a cycle from creating duplicate searches. For this problem, hold onto the concrete trace: each unseen node starts a flood and increments the component count.</figcaption></figure>

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
