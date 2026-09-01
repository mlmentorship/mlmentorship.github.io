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
<figure class="learning-figure coding-visual-figure" aria-labelledby="number-of-connected-components-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="number-of-connected-components-state-title">Number of Connected Components: Every unseen node starts one DFS component and marks its whole group.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="number-of-connected-components" role="group" tabindex="0" aria-label="Number of Connected Components: Every unseen node starts one DFS component and marks its whole group."><div class="coding-visual-example"><span>Input and goal</span><strong>Count separate groups in an undirected graph.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Start component 1"><div class="coding-trace-frame-heading"><span>Start component 1</span><strong>Node 0 reaches 1 and 2.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0-1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="401.67960777017606" y2="90.8966744387541" /><text x="320.83980388508803" y="57.94833721937705">0-1</text></g><g data-motion-key="edge-1-2-1"><line class="coding-trace-edge-line" x1="401.67960777017606" y1="90.8966744387541" x2="339.9234928897204" y2="178.1033255612459" /><text x="370.8015503299482" y="128.5">1-2</text></g><g data-motion-key="edge-3-4-2"><line class="coding-trace-edge-line" x1="140.07650711027958" y1="178.1033255612459" x2="78.32039222982388" y2="90.8966744387541" /><text x="109.19844967005173" y="128.5">3-4</text></g><g class="coding-trace-graph-node is-state" data-motion-key="node-0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">0</text></g><g class="coding-trace-graph-node is-state" data-motion-key="node-1-1"><circle cx="401.67960777017606" cy="90.8966744387541" r="23" /><text x="401.67960777017606" y="94.8966744387541">1</text></g><g class="coding-trace-graph-node is-state" data-motion-key="node-2-2"><circle cx="339.9234928897204" cy="178.1033255612459" r="23" /><text x="339.9234928897204" y="182.1033255612459">2</text></g><g class="coding-trace-graph-node" data-motion-key="node-3-3"><circle cx="140.07650711027958" cy="178.1033255612459" r="23" /><text x="140.07650711027958" y="182.1033255612459">3</text></g><g class="coding-trace-graph-node" data-motion-key="node-4-4"><circle cx="78.32039222982388" cy="90.8966744387541" r="23" /><text x="78.32039222982388" y="94.8966744387541">4</text></g></svg><div class="coding-trace-meta"><span><b>visited</b>0, 1, 2</span><span><b>components</b>1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Find the next unseen node"><div class="coding-trace-frame-heading"><span>Find the next unseen node</span><strong>Node 3 starts a second flood and reaches 4.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0-1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="401.67960777017606" y2="90.8966744387541" /><text x="320.83980388508803" y="57.94833721937705">0-1</text></g><g data-motion-key="edge-1-2-1"><line class="coding-trace-edge-line" x1="401.67960777017606" y1="90.8966744387541" x2="339.9234928897204" y2="178.1033255612459" /><text x="370.8015503299482" y="128.5">1-2</text></g><g data-motion-key="edge-3-4-2"><line class="coding-trace-edge-line" x1="140.07650711027958" y1="178.1033255612459" x2="78.32039222982388" y2="90.8966744387541" /><text x="109.19844967005173" y="128.5">3-4</text></g><g class="coding-trace-graph-node is-state" data-motion-key="node-0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">0</text></g><g class="coding-trace-graph-node is-state" data-motion-key="node-1-1"><circle cx="401.67960777017606" cy="90.8966744387541" r="23" /><text x="401.67960777017606" y="94.8966744387541">1</text></g><g class="coding-trace-graph-node is-state" data-motion-key="node-2-2"><circle cx="339.9234928897204" cy="178.1033255612459" r="23" /><text x="339.9234928897204" y="182.1033255612459">2</text></g><g class="coding-trace-graph-node is-state" data-motion-key="node-3-3"><circle cx="140.07650711027958" cy="178.1033255612459" r="23" /><text x="140.07650711027958" y="182.1033255612459">3</text></g><g class="coding-trace-graph-node is-state" data-motion-key="node-4-4"><circle cx="78.32039222982388" cy="90.8966744387541" r="23" /><text x="78.32039222982388" y="94.8966744387541">4</text></g></svg><div class="coding-trace-meta"><span><b>visited</b>0, 1, 2, 3, 4</span><span><b>components</b>2</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the count"><div class="coding-trace-frame-heading"><span>Return the count</span><strong>Two starting floods mean two connected components.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0-1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="401.67960777017606" y2="90.8966744387541" /><text x="320.83980388508803" y="57.94833721937705">0-1</text></g><g data-motion-key="edge-1-2-1"><line class="coding-trace-edge-line" x1="401.67960777017606" y1="90.8966744387541" x2="339.9234928897204" y2="178.1033255612459" /><text x="370.8015503299482" y="128.5">1-2</text></g><g data-motion-key="edge-3-4-2"><line class="coding-trace-edge-line" x1="140.07650711027958" y1="178.1033255612459" x2="78.32039222982388" y2="90.8966744387541" /><text x="109.19844967005173" y="128.5">3-4</text></g><g class="coding-trace-graph-node" data-motion-key="node-0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">0</text></g><g class="coding-trace-graph-node" data-motion-key="node-1-1"><circle cx="401.67960777017606" cy="90.8966744387541" r="23" /><text x="401.67960777017606" y="94.8966744387541">1</text></g><g class="coding-trace-graph-node" data-motion-key="node-2-2"><circle cx="339.9234928897204" cy="178.1033255612459" r="23" /><text x="339.9234928897204" y="182.1033255612459">2</text></g><g class="coding-trace-graph-node" data-motion-key="node-3-3"><circle cx="140.07650711027958" cy="178.1033255612459" r="23" /><text x="140.07650711027958" y="182.1033255612459">3</text></g><g class="coding-trace-graph-node" data-motion-key="node-4-4"><circle cx="78.32039222982388" cy="90.8966744387541" r="23" /><text x="78.32039222982388" y="94.8966744387541">4</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>2</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start component 1</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Find the next unseen node</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the count</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Every unseen node starts one DFS component and marks its whole group.</p></div><figcaption><strong>Read it this way:</strong> Node 0 reaches 1 and 2. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
