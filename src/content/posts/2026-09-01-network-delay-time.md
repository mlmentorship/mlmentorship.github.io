---
title: "Network Delay Time"
description: "Find when a signal from one node reaches every node in a weighted directed graph."
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

> Find when a signal from one node reaches every node in a weighted directed graph.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:network-delay-time-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="network-delay-time-state-title"><p class="visual-kicker">The cheapest frontier first</p><p class="visual-title" id="network-delay-time-state-title">Network Delay Time: Finalize a node when no cheaper path remains</p><div class="coding-visual coding-visual--dijkstra" data-coding-visual data-coding-mode="dijkstra" data-coding-slug="network-delay-time" role="group" aria-label="Network Delay Time: paths 1-&gt;2 cost 1 and 1-&gt;3 cost 4 -&gt; finalize 2 before 3. Every finalized node has the smallest possible distance from the source."><div class="coding-visual-example"><span>Concrete trace</span><strong>paths 1-&gt;2 cost 1 and 1-&gt;3 cost 4 -&gt; finalize 2 before 3</strong></div><div class="coding-visual-sketch coding-visual-sketch--dijkstra"><div class="coding-sketch-path"><span class="coding-sketch-node coding-sketch-node--active">start</span><span class="coding-sketch-edge">cost 1</span><span class="coding-sketch-node">next</span><span class="coding-sketch-edge">cost 4</span><span class="coding-sketch-node">farther</span></div><p class="coding-sketch-note">compare total path cost, then lock the cheapest frontier node</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Seed</span><strong>distance 0</strong><small>Start with the source and its known cost.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Choose</span><strong>min heap</strong><small>Pop the reachable path with least total cost.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Relax</span><strong>new cost</strong><small>Offer each outgoing path if it improves the estimate.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Finalize</span><strong>locked distance</strong><small>The popped cost is final when edges are nonnegative.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every finalized node has the smallest possible distance from the source.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The heap orders paths by total cost, not by the last edge. Once the cheapest frontier path reaches a node, any alternative must be at least as expensive. For this problem, hold onto the concrete trace: paths 1-&gt;2 cost 1 and 1-&gt;3 cost 4 -&gt; finalize 2 before 3.</figcaption></figure>

**Pattern:** Dijkstra.

**Simple idea:** Always process the path with the lowest total cost next. Add each outgoing
edge cost and put the new path in the heap.

```python
import heapq
from collections import defaultdict

def network_delay_time(times: list[list[int]], node_count: int, start: int) -> int:
   graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
   for source, target, cost in times:
      graph[source].append((target, cost))

   distances: dict[int, int] = {}
   heap = [(0, start)]

   while heap:
      distance, node = heapq.heappop(heap)
      if node in distances:
         continue

      distances[node] = distance
      for neighbor, cost in graph[node]:
         if neighbor not in distances:
            heapq.heappush(heap, (distance + cost, neighbor))

   return max(distances.values()) if len(distances) == node_count else -1
```

**Cost:** $O((V + E)\log V)$ time and $O(V + E)$ space.
