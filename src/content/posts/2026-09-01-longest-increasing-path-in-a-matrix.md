---
title: "Longest Increasing Path in a Matrix"
description: "Use strict value growth to justify a coordinate-only suffix cache, then change evaluation order to avoid recursion or count all increasing paths."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

Follow only arrows toward larger values. A route cannot return to an earlier cell, so every caller can reuse that cell's longest continuation.

## Problem

> Find the longest path that moves to a larger neighboring value each step.

Use an unchanged rectangular integer matrix and four orthogonal neighbors. Count cells, not edges. Empty input returns zero.

## Picture

For `[[3,4,5],[2,1,6]]`, draw an arrow from each cell to each larger orthogonal neighbor.
The longest route follows values `1,2,3,4,5,6`, using six distinct cells.

A cycle would require strict increases back to the starting value, making that value larger than itself.
Thus no path-local visited set is needed. A continuation cannot revisit its caller's smaller-valued prefix.

Define $L(r,c)$ as the longest increasing path **starting** at coordinate $(r,c)$, including that cell.
A peak returns one. Otherwise choose the largest $1+L$ among its larger neighbors.

| Completed return | Cached length | Reason |
| --- | --- | --- |
| `(1,2)`, value 6 | 1 | No larger neighbor |
| `(0,2)`, value 5 | 2 | One plus 6's suffix |
| `(0,1)`, value 4 | 3 | One plus 5's suffix |
| `(0,0)`, value 3 | 4 | One plus 4's suffix |
| `(1,0)`, value 2 | 5 | Reuse 3's completed suffix |
| `(1,1)`, value 1 | 6 | Best among neighbors 4, 6 and 2 |

The memo becomes `[[4,3,2],[5,6,1]]`. A cached value means a completed answer, not merely a discovered cell.
Global flood-fill marks would wrongly suppress valid reuse by other callers.

Coordinate alone identifies the suffix because the immutable matrix fixes every legal continuation.
Create the cache inside each call. Sharing it across changed matrices would reuse answers from the wrong graph.

Equal-valued neighbors have no edge. Changing `>` to `>=` permits cycles and invalidates this reasoning.

<!-- visual:longest-increasing-path-in-a-matrix-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-increasing-path-in-a-matrix-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-increasing-path-in-a-matrix-state-title">Longest Increasing Path in a Matrix: Treat larger-neighbor moves as a DAG, recursively solve each cell once, and memoize the longest suffix length returned to its predecessors.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-increasing-path-in-a-matrix" role="group" tabindex="0" aria-label="Longest Increasing Path in a Matrix: Treat larger-neighbor moves as a DAG, recursively solve each cell once, and memoize the longest suffix length returned to its predecessors."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the longest path that moves to a larger neighboring value each step.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="start-3" role="group" aria-label="Start DFS at value 3"><div class="coding-trace-frame-heading"><span>Start DFS at value 3</span><strong>The outer max begins at (0,0)=3. Only larger neighbor 4 can extend this path.</strong></div><div class="coding-trace-graph is-tree-layout"><svg viewBox="0 0 350 235" role="img" aria-label="Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks"><defs><marker id="increasing-path-0-0-0" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="71" y1="45" x2="146" y2="45" marker-end="url(#increasing-path-0-0-0)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="45" x2="276" y2="45" marker-end="url(#increasing-path-0-0-0)" /></g><g><line class="coding-trace-edge-line" x1="305" y1="71" x2="305" y2="156" marker-end="url(#increasing-path-0-0-0)" /></g><g><line class="coding-trace-edge-line" x1="45" y1="159" x2="45" y2="74" marker-end="url(#increasing-path-0-0-0)" /></g><g><line class="coding-trace-edge-line" x1="175" y1="159" x2="175" y2="74" marker-end="url(#increasing-path-0-0-0)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="185" x2="276" y2="185" marker-end="url(#increasing-path-0-0-0)" /></g><g><line class="coding-trace-edge-line" x1="149" y1="185" x2="74" y2="185" marker-end="url(#increasing-path-0-0-0)" /></g><g class="coding-trace-graph-node is-focus"><circle cx="45" cy="45" r="23" /><text x="45" y="49">3</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="45" r="23" /><text x="175" y="49">4</text></g><g class="coding-trace-graph-node"><circle cx="305" cy="45" r="23" /><text x="305" y="49">5</text></g><g class="coding-trace-graph-node"><circle cx="45" cy="185" r="23" /><text x="45" y="189">2</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="185" r="23" /><text x="175" y="189">1</text></g><g class="coding-trace-graph-node"><circle cx="305" cy="185" r="23" /><text x="305" y="189">6</text></g></svg></div><div class="coding-trace-meta"><span><b>completed</b>0</span></div><details><summary>Original matrix and DFS cursor</summary><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="dfs-cursor"><span>3</span><small>DFS cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span></div></details><div class="coding-trace-meta"><span><b>memo</b>[.,.,.; .,.,.]</span><span><b>dependency</b>3 -&gt; 4</span><span><b>best</b>path_from(0,0) starts at 1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="descend-5" hidden role="group" aria-label="Descend through 4 and 5"><div class="coding-trace-frame-heading"><span>Descend through 4 and 5</span><strong>At 4, only 5 is larger. At 5, the down neighbor 6 is larger.</strong></div><div class="coding-trace-graph is-tree-layout"><svg viewBox="0 0 350 235" role="img" aria-label="Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks"><defs><marker id="increasing-path-0-2-0" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="71" y1="45" x2="146" y2="45" marker-end="url(#increasing-path-0-2-0)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="45" x2="276" y2="45" marker-end="url(#increasing-path-0-2-0)" /></g><g><line class="coding-trace-edge-line" x1="305" y1="71" x2="305" y2="156" marker-end="url(#increasing-path-0-2-0)" /></g><g><line class="coding-trace-edge-line" x1="45" y1="159" x2="45" y2="74" marker-end="url(#increasing-path-0-2-0)" /></g><g><line class="coding-trace-edge-line" x1="175" y1="159" x2="175" y2="74" marker-end="url(#increasing-path-0-2-0)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="185" x2="276" y2="185" marker-end="url(#increasing-path-0-2-0)" /></g><g><line class="coding-trace-edge-line" x1="149" y1="185" x2="74" y2="185" marker-end="url(#increasing-path-0-2-0)" /></g><g class="coding-trace-graph-node"><circle cx="45" cy="45" r="23" /><text x="45" y="49">3</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="45" r="23" /><text x="175" y="49">4</text></g><g class="coding-trace-graph-node is-focus"><circle cx="305" cy="45" r="23" /><text x="305" y="49">5</text></g><g class="coding-trace-graph-node"><circle cx="45" cy="185" r="23" /><text x="45" y="189">2</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="185" r="23" /><text x="175" y="189">1</text></g><g class="coding-trace-graph-node"><circle cx="305" cy="185" r="23" /><text x="305" y="189">6</text></g></svg></div><div class="coding-trace-meta"><span><b>completed</b>0</span></div><details><summary>Original matrix and DFS cursor</summary><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="dfs-cursor"><span>5</span><small>DFS cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span></div></details><div class="coding-trace-meta"><span><b>memo</b>[.,.,.; .,.,.]</span><span><b>callStack</b>3 -&gt; 4 -&gt; 5</span><span><b>dependency</b>5 -&gt; 6</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="cache-6" hidden role="group" aria-label="Cache the peak 6"><div class="coding-trace-frame-heading"><span>Cache the peak 6</span><strong>Value 6 has no larger neighbor, so path_from(1,2)=1.</strong></div><div class="coding-trace-graph is-tree-layout"><svg viewBox="0 0 350 235" role="img" aria-label="Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks"><defs><marker id="increasing-path-1-2-1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="71" y1="45" x2="146" y2="45" marker-end="url(#increasing-path-1-2-1)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="45" x2="276" y2="45" marker-end="url(#increasing-path-1-2-1)" /></g><g><line class="coding-trace-edge-line" x1="305" y1="71" x2="305" y2="156" marker-end="url(#increasing-path-1-2-1)" /></g><g><line class="coding-trace-edge-line" x1="45" y1="159" x2="45" y2="74" marker-end="url(#increasing-path-1-2-1)" /></g><g><line class="coding-trace-edge-line" x1="175" y1="159" x2="175" y2="74" marker-end="url(#increasing-path-1-2-1)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="185" x2="276" y2="185" marker-end="url(#increasing-path-1-2-1)" /></g><g><line class="coding-trace-edge-line" x1="149" y1="185" x2="74" y2="185" marker-end="url(#increasing-path-1-2-1)" /></g><g class="coding-trace-graph-node"><circle cx="45" cy="45" r="23" /><text x="45" y="49">3</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="45" r="23" /><text x="175" y="49">4</text></g><g class="coding-trace-graph-node"><circle cx="305" cy="45" r="23" /><text x="305" y="49">5</text></g><g class="coding-trace-graph-node"><circle cx="45" cy="185" r="23" /><text x="45" y="189">2</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="185" r="23" /><text x="175" y="189">1</text></g><g class="coding-trace-graph-node is-focus"><circle cx="305" cy="185" r="23" /><text x="305" y="189">6</text></g></svg></div><div class="coding-trace-meta"><span><b>completed</b>1</span></div><details><summary>Original matrix and DFS cursor</summary><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="dfs-cursor"><span>6</span><small>DFS cursor</small></span></div></details><div class="coding-trace-meta"><span><b>memo</b>[.,.,.; .,.,1]</span><span><b>arithmetic</b>best(6) = 1</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="cache-5" hidden role="group" aria-label="Return and cache 5"><div class="coding-trace-frame-heading"><span>Return and cache 5</span><strong>The recursive result from 6 gives best(5)=max(1,1+1)=2.</strong></div><div class="coding-trace-graph is-tree-layout"><svg viewBox="0 0 350 235" role="img" aria-label="Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks"><defs><marker id="increasing-path-0-2-2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="71" y1="45" x2="146" y2="45" marker-end="url(#increasing-path-0-2-2)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="45" x2="276" y2="45" marker-end="url(#increasing-path-0-2-2)" /></g><g><line class="coding-trace-edge-line" x1="305" y1="71" x2="305" y2="156" marker-end="url(#increasing-path-0-2-2)" /></g><g><line class="coding-trace-edge-line" x1="45" y1="159" x2="45" y2="74" marker-end="url(#increasing-path-0-2-2)" /></g><g><line class="coding-trace-edge-line" x1="175" y1="159" x2="175" y2="74" marker-end="url(#increasing-path-0-2-2)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="185" x2="276" y2="185" marker-end="url(#increasing-path-0-2-2)" /></g><g><line class="coding-trace-edge-line" x1="149" y1="185" x2="74" y2="185" marker-end="url(#increasing-path-0-2-2)" /></g><g class="coding-trace-graph-node"><circle cx="45" cy="45" r="23" /><text x="45" y="49">3</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="45" r="23" /><text x="175" y="49">4</text></g><g class="coding-trace-graph-node is-focus"><circle cx="305" cy="45" r="23" /><text x="305" y="49">5</text></g><g class="coding-trace-graph-node"><circle cx="45" cy="185" r="23" /><text x="45" y="189">2</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="185" r="23" /><text x="175" y="189">1</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="185" r="23" /><text x="305" y="189">6</text></g></svg></div><div class="coding-trace-meta"><span><b>completed</b>2</span></div><details><summary>Original matrix and DFS cursor</summary><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="dfs-cursor"><span>5</span><small>DFS cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span></div></details><div class="coding-trace-meta"><span><b>memo</b>[.,.,2; .,.,1]</span><span><b>arithmetic</b>L(0,2) = max(1, 1 + memo[1,2]) = 2</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="cache-3" hidden role="group" aria-label="Return through 4 and 3"><div class="coding-trace-frame-heading"><span>Return through 4 and 3</span><strong>Cache best(4)=1+2=3, then best(3)=1+3=4.</strong></div><div class="coding-trace-graph is-tree-layout"><svg viewBox="0 0 350 235" role="img" aria-label="Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks"><defs><marker id="increasing-path-0-0-4" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="71" y1="45" x2="146" y2="45" marker-end="url(#increasing-path-0-0-4)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="45" x2="276" y2="45" marker-end="url(#increasing-path-0-0-4)" /></g><g><line class="coding-trace-edge-line" x1="305" y1="71" x2="305" y2="156" marker-end="url(#increasing-path-0-0-4)" /></g><g><line class="coding-trace-edge-line" x1="45" y1="159" x2="45" y2="74" marker-end="url(#increasing-path-0-0-4)" /></g><g><line class="coding-trace-edge-line" x1="175" y1="159" x2="175" y2="74" marker-end="url(#increasing-path-0-0-4)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="185" x2="276" y2="185" marker-end="url(#increasing-path-0-0-4)" /></g><g><line class="coding-trace-edge-line" x1="149" y1="185" x2="74" y2="185" marker-end="url(#increasing-path-0-0-4)" /></g><g class="coding-trace-graph-node is-focus"><circle cx="45" cy="45" r="23" /><text x="45" y="49">3</text></g><g class="coding-trace-graph-node is-state"><circle cx="175" cy="45" r="23" /><text x="175" y="49">4</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="45" r="23" /><text x="305" y="49">5</text></g><g class="coding-trace-graph-node"><circle cx="45" cy="185" r="23" /><text x="45" y="189">2</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="185" r="23" /><text x="175" y="189">1</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="185" r="23" /><text x="305" y="189">6</text></g></svg></div><div class="coding-trace-meta"><span><b>completed</b>4</span></div><details><summary>Original matrix and DFS cursor</summary><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="dfs-cursor"><span>3</span><small>DFS cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span></div></details><div class="coding-trace-meta"><span><b>memo</b>[4,3,2; .,.,1]</span><span><b>arithmetic</b>best(4)=3; best(3)=4</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="cache-2" hidden role="group" aria-label="Solve value 2 with a cache hit"><div class="coding-trace-frame-heading"><span>Solve value 2 with a cache hit</span><strong>The outer scan reaches (1,0)=2. Its larger neighbor 3 is cached at 4, so best(2)=1+4=5.</strong></div><div class="coding-trace-graph is-tree-layout"><svg viewBox="0 0 350 235" role="img" aria-label="Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks"><defs><marker id="increasing-path-1-0-5" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="71" y1="45" x2="146" y2="45" marker-end="url(#increasing-path-1-0-5)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="45" x2="276" y2="45" marker-end="url(#increasing-path-1-0-5)" /></g><g><line class="coding-trace-edge-line" x1="305" y1="71" x2="305" y2="156" marker-end="url(#increasing-path-1-0-5)" /></g><g><line class="coding-trace-edge-line" x1="45" y1="159" x2="45" y2="74" marker-end="url(#increasing-path-1-0-5)" /></g><g><line class="coding-trace-edge-line" x1="175" y1="159" x2="175" y2="74" marker-end="url(#increasing-path-1-0-5)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="185" x2="276" y2="185" marker-end="url(#increasing-path-1-0-5)" /></g><g><line class="coding-trace-edge-line" x1="149" y1="185" x2="74" y2="185" marker-end="url(#increasing-path-1-0-5)" /></g><g class="coding-trace-graph-node is-state"><circle cx="45" cy="45" r="23" /><text x="45" y="49">3</text></g><g class="coding-trace-graph-node is-state"><circle cx="175" cy="45" r="23" /><text x="175" y="49">4</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="45" r="23" /><text x="305" y="49">5</text></g><g class="coding-trace-graph-node is-focus"><circle cx="45" cy="185" r="23" /><text x="45" y="189">2</text></g><g class="coding-trace-graph-node"><circle cx="175" cy="185" r="23" /><text x="175" y="189">1</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="185" r="23" /><text x="305" y="189">6</text></g></svg></div><div class="coding-trace-meta"><span><b>completed</b>5</span></div><details><summary>Original matrix and DFS cursor</summary><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>5</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="dfs-cursor"><span>2</span><small>DFS cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span></div></details><div class="coding-trace-meta"><span><b>memo</b>[4,3,2; 5,.,1]</span><span><b>cacheHit</b>memo[0,0] = 4</span><span><b>arithmetic</b>best(2) = max(1, 1 + 4) = 5</span></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="cache-1" hidden role="group" aria-label="Solve the three choices at 1"><div class="coding-trace-frame-heading"><span>Solve the three choices at 1</span><strong>Value 1 can extend through 4, 6 or 2. Their candidate lengths are four, two and six. Choose six.</strong></div><div class="coding-trace-graph is-tree-layout"><svg viewBox="0 0 350 235" role="img" aria-label="Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks"><defs><marker id="increasing-path-1-1-6" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="71" y1="45" x2="146" y2="45" marker-end="url(#increasing-path-1-1-6)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="45" x2="276" y2="45" marker-end="url(#increasing-path-1-1-6)" /></g><g><line class="coding-trace-edge-line" x1="305" y1="71" x2="305" y2="156" marker-end="url(#increasing-path-1-1-6)" /></g><g><line class="coding-trace-edge-line" x1="45" y1="159" x2="45" y2="74" marker-end="url(#increasing-path-1-1-6)" /></g><g><line class="coding-trace-edge-line" x1="175" y1="159" x2="175" y2="74" marker-end="url(#increasing-path-1-1-6)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="185" x2="276" y2="185" marker-end="url(#increasing-path-1-1-6)" /></g><g><line class="coding-trace-edge-line" x1="149" y1="185" x2="74" y2="185" marker-end="url(#increasing-path-1-1-6)" /></g><g class="coding-trace-graph-node is-state"><circle cx="45" cy="45" r="23" /><text x="45" y="49">3</text></g><g class="coding-trace-graph-node is-state"><circle cx="175" cy="45" r="23" /><text x="175" y="49">4</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="45" r="23" /><text x="305" y="49">5</text></g><g class="coding-trace-graph-node is-state"><circle cx="45" cy="185" r="23" /><text x="45" y="189">2</text></g><g class="coding-trace-graph-node is-focus"><circle cx="175" cy="185" r="23" /><text x="175" y="189">1</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="185" r="23" /><text x="305" y="189">6</text></g></svg></div><div class="coding-trace-meta"><span><b>completed</b>6</span></div><details><summary>Original matrix and DFS cursor</summary><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="dfs-cursor"><span>1</span><small>DFS cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span></div></details><div class="coding-trace-meta"><span><b>memo</b>[4,3,2; 5,6,1]</span><span><b>dependencies</b>1 -&gt; 4, 1 -&gt; 6, 1 -&gt; 2</span><span><b>arithmetic</b>max(1, 1+3, 1+1, 1+5) = 6</span></div></div><div class="coding-trace-frame" data-coding-frame="7" data-frame-key="finish" hidden role="group" aria-label="Take the maximum over all starts"><div class="coding-trace-frame-heading"><span>Take the maximum over all starts</span><strong>All six cells are memoized once. The maximum length is six, following values 1,2,3,4,5,6.</strong></div><div class="coding-trace-graph is-tree-layout"><svg viewBox="0 0 350 235" role="img" aria-label="Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks"><defs><marker id="increasing-path-1-1-6-1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="71" y1="45" x2="146" y2="45" marker-end="url(#increasing-path-1-1-6-1)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="45" x2="276" y2="45" marker-end="url(#increasing-path-1-1-6-1)" /></g><g><line class="coding-trace-edge-line" x1="305" y1="71" x2="305" y2="156" marker-end="url(#increasing-path-1-1-6-1)" /></g><g><line class="coding-trace-edge-line" x1="45" y1="159" x2="45" y2="74" marker-end="url(#increasing-path-1-1-6-1)" /></g><g><line class="coding-trace-edge-line" x1="175" y1="159" x2="175" y2="74" marker-end="url(#increasing-path-1-1-6-1)" /></g><g><line class="coding-trace-edge-line" x1="201" y1="185" x2="276" y2="185" marker-end="url(#increasing-path-1-1-6-1)" /></g><g><line class="coding-trace-edge-line" x1="149" y1="185" x2="74" y2="185" marker-end="url(#increasing-path-1-1-6-1)" /></g><g class="coding-trace-graph-node is-state"><circle cx="45" cy="45" r="23" /><text x="45" y="49">3</text></g><g class="coding-trace-graph-node is-state"><circle cx="175" cy="45" r="23" /><text x="175" y="49">4</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="45" r="23" /><text x="305" y="49">5</text></g><g class="coding-trace-graph-node is-state"><circle cx="45" cy="185" r="23" /><text x="45" y="189">2</text></g><g class="coding-trace-graph-node is-focus"><circle cx="175" cy="185" r="23" /><text x="175" y="189">1</text></g><g class="coding-trace-graph-node is-state"><circle cx="305" cy="185" r="23" /><text x="305" y="189">6</text></g></svg></div><div class="coding-trace-meta"><span><b>completed</b>6</span></div><details><summary>Original matrix and DFS cursor</summary><div class="coding-trace-grid" style="--trace-cols:3" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>2</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="dfs-cursor"><span>1</span><small>DFS cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>6</span></span></div></details><div class="coding-trace-meta"><span><b>memo</b>[4,3,2; 5,6,1]</span><span><b>path</b>(1,1)-&gt;(1,0)-&gt;(0,0)-&gt;(0,1)-&gt;(0,2)-&gt;(1,2)</span><span><b>result</b>6</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 8</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start DFS at value 3</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Descend through 4 and 5</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Cache the peak 6</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Return and cache 5</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Return through 4 and 3</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Solve value 2 with a cache hit</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Solve the three choices at 1</strong></button><button type="button" data-coding-frame-button="7"><span>8</span><strong>Take the maximum over all starts</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use memoized DFS when every cell asks for an optimal path through neighboring states and a strict monotone move rule prevents cycles.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Once path_from(r,c) returns, its memo value is the longest increasing path starting at that cell; every candidate is 1 plus an already correct recursively computed larger-neighbor suffix.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">A strict ranking function can turn implicit neighbor moves into a DAG; memoize the optimal suffix at each state and combine it from higher-ranked neighbors.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> The outer max begins at (0,0)=3. Only larger neighbor 4 can extend this path. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

```python
from functools import cache

def longest_increasing_path(matrix: list[list[int]]) -> int:
   if not matrix or not matrix[0]:
      return 0

   @cache
   def path_from(row: int, col: int) -> int:
      best = 1
      for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
         new_row = row + row_step
         new_col = col + col_step
         if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
            if matrix[new_row][new_col] > matrix[row][col]:
               best = max(best, 1 + path_from(new_row, new_col))
      return best

   return max(
      path_from(row, col)
      for row in range(len(matrix))
      for col in range(len(matrix[0]))
   )
```

**Cost:** $O(rows \times cols)$ time and space.

Each coordinate computes once and checks four neighbors. Cache size and maximum recursive depth can both reach $O(rows\cdot cols)$.
Python can raise `RecursionError` on a long valid chain; linear total work does not prevent that failure.

<!-- learning:core-end -->

## Change Evaluation or Aggregation

Start from `longest_increasing_path`. Preserve strict growth while changing how completed states are obtained or combined.

1. **Avoid recursive depth.**

   **New task:** Support increasing chains longer than Python's call-stack limit.

   **Picture and change:** Count incoming edges from smaller neighbors. Use [zero-indegree elimination](/questions/course-schedule/) in frozen batches.
   A cell becomes ready after its last predecessor, so its layer is one plus the maximum predecessor layer.
   This measures length ending there; the global maximum matches the source's starting-length maximum.

   **Code:** Complete replacement:

   ```python
   from collections import deque


   def increasing_path_layers(matrix):
      if not matrix or not matrix[0]:
         return 0
      rows, columns = len(matrix), len(matrix[0])
      steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
      degree = [[0] * columns for _ in range(rows)]
      for row in range(rows):
         for column in range(columns):
            degree[row][column] = sum(
               0 <= row + row_step < rows
               and 0 <= column + col_step < columns
               and matrix[row + row_step][column + col_step] < matrix[row][column]
               for row_step, col_step in steps
            )
      frontier = deque(
         (row, column)
         for row in range(rows)
         for column in range(columns)
         if degree[row][column] == 0
      )
      layers = 0
      while frontier:
         layers += 1
         for _ in range(len(frontier)):
            row, column = frontier.popleft()
            for row_step, col_step in steps:
               new_row, new_col = row + row_step, column + col_step
               if 0 <= new_row < rows and 0 <= new_col < columns:
                  if matrix[new_row][new_col] > matrix[row][column]:
                     degree[new_row][new_col] -= 1
                     if degree[new_row][new_col] == 0:
                        frontier.append((new_row, new_col))
      return layers
   ```

   **Check:** The six-cell example returns six. `[[1,1]]` returns one, not two.

   **Use when:** Recursion depth is a risk. Work and degree/frontier storage remain $O(rows\cdot cols)$ without recursive calls.

2. **Count every increasing path.**

   **New task:** Count all nonempty coordinate paths, not just the longest length.

   **Picture and change:** Process cells by increasing value. Each has one singleton path plus paths ending at its smaller neighbors.
   Sum those ending counts. Equal values create no dependency, regardless of their sorting order.

   **Code:** Complete replacement:

   ```python
   def count_increasing_paths(matrix):
      if not matrix or not matrix[0]:
         return 0
      rows, columns = len(matrix), len(matrix[0])
      counts = [[0] * columns for _ in range(rows)]
      cells = sorted(
         (matrix[row][column], row, column)
         for row in range(rows)
         for column in range(columns)
      )
      for value, row, column in cells:
         counts[row][column] = 1
         for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_row, new_col = row + row_step, column + col_step
            if 0 <= new_row < rows and 0 <= new_col < columns:
               if matrix[new_row][new_col] < value:
                  counts[row][column] += counts[new_row][new_col]
      return sum(map(sum, counts))
   ```

   **Check:** `[[1,2]]` has three paths: `[1]`, `[2]`, `[1,2]`. `[[1,1]]` has two singletons.

   **Use when:** Multiplicity replaces length. For $V=rows\cdot cols$, sorting costs $O(V\log(V+1))$; storage is $O(V)$.
   Exact counts can grow large, so arithmetic cost also depends on integer bit lengths.

## Rebuild State Sufficiency

Why can another caller reuse the same coordinate's suffix?

**Answer:** Strict growth prevents reentering its smaller prefix. The unchanged matrix, not the caller's route, determines every legal continuation.
