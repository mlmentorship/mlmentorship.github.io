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
<figure class="learning-figure coding-visual-figure" aria-labelledby="network-delay-time-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="network-delay-time-state-title">Network Delay Time: Dijkstra finalizes the node whose total path cost is smallest.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="network-delay-time" role="group" aria-label="Network Delay Time: Dijkstra finalizes the node whose total path cost is smallest."><div class="coding-visual-example"><span>Input and goal</span><strong>Find when a signal from one node reaches every node in a weighted directed graph.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Start at node 2"><div class="coding-trace-frame-heading"><span>Start at node 2</span><strong>Known distance is 0. Its outgoing paths cost 1 to nodes 1 and 3.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node">1</span><span class="coding-trace-node is-focus is-state">2</span><span class="coding-trace-node">3</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">2 -1-&gt; 1</span><span class="coding-trace-edge">2 -1-&gt; 3</span></div><div class="coding-trace-meta"><span><b>visited</b>2:0</span><span><b>frontier</b>1:1, 3:1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Finalize the cheapest path"><div class="coding-trace-frame-heading"><span>Finalize the cheapest path</span><strong>Pop node 1 at distance 1; node 3 is already reachable at distance 1.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node is-state">1</span><span class="coding-trace-node is-state">2</span><span class="coding-trace-node">3</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">2 -1-&gt; 1</span><span class="coding-trace-edge">2 -1-&gt; 3</span><span class="coding-trace-edge">1 -1-&gt; 3</span></div><div class="coding-trace-meta"><span><b>visited</b>2:0, 1:1</span><span><b>frontier</b>3:1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Take the farthest finalized distance"><div class="coding-trace-frame-heading"><span>Take the farthest finalized distance</span><strong>Every node is reached; the delay is max(0,1,1) = 1.</strong></div><div class="coding-trace-graph"><div class="coding-trace-node-row"><span class="coding-trace-node is-state">1</span><span class="coding-trace-node is-state">2</span><span class="coding-trace-node is-state">3</span></div><div class="coding-trace-edge-row"><span class="coding-trace-edge">2 -1-&gt; 1</span><span class="coding-trace-edge">2 -1-&gt; 3</span></div><div class="coding-trace-meta"><span><b>visited</b>2:0, 1:1, 3:1</span></div></div><div class="coding-trace-meta"><span><b>result</b>1</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start at node 2</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Finalize the cheapest path</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Take the farthest finalized distance</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Dijkstra finalizes the node whose total path cost is smallest.</p></div><figcaption><strong>Read it this way:</strong> Known distance is 0. Its outgoing paths cost 1 to nodes 1 and 3. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
