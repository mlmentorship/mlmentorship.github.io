---
title: "Allocate one copy per original identity before following edges"
description: "Memoize copies before discovery, then preserve every ordered adjacency entry across cycles, shared nodes and multiple roots."
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

Allocate and memoize each copy before following its edges; resolve every copied neighbor through that identity map.

## Problem

Copy the component reachable from one `GraphNode`, preserving integer values and adjacency topology.
In the triangle, original nodes `O1`, `O2`, and `O3` must become separate copies `C1`, `C2`, and `C3`.
Every copied edge must stay inside the copies.
Preserve neighbor order, repeated entries, shared targets, and self-loops.
`None` returns `None`; unreachable components are not copied.

## Picture

Before processing `O1`, create `C1` and record `O1 -> C1`.
When `O1` discovers `O2` and `O3`, allocate each copy once and enqueue each original once.
Back edges find existing copies; they do not restart copying.

The memo's keys are **object identities**, not numeric labels.
Two different nodes labelled `7` still require two different copies.

<!-- visual:clone-graph-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="clone-graph-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="clone-graph-state-title">Clone Graph: Create one mapped copy per original node, then reproduce every adjacency entry between copies.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="clone-graph" role="group" tabindex="0" aria-label="Clone Graph: Create one mapped copy per original node, then reproduce every adjacency entry between copies."><div class="coding-visual-example"><span>Input and goal</span><strong>Make a deep copy of a connected graph.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="seed-copy-map" role="group" aria-label="Create the first copy"><div class="coding-trace-frame-heading"><span>Create the first copy</span><strong>For triangle O1-O2-O3, create C1 before traversal and enqueue O1. The map is {O1:C1}.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><defs><marker id="coding-graph-arrow-7" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-graph-region" x="90" y="20">original</text><text class="coding-trace-graph-region" x="390" y="20">clone</text><g data-motion-key="edge-O1 &lt;-&gt; O2-0"><line class="coding-trace-edge-line" x1="80.05885338911621" y1="82.02443764296916" x2="53.08820198906269" y2="147.20351185976517" /></g><g data-motion-key="edge-O1 &lt;-&gt; O3-1"><line class="coding-trace-edge-line" x1="99.94114661088379" y1="82.02443764296916" x2="126.9117980109373" y2="147.20351185976517" /></g><g data-motion-key="edge-O2 &lt;-&gt; O3-2"><line class="coding-trace-edge-line" x1="68" y1="174" x2="109" y2="174" /></g><g class="coding-trace-graph-node is-focus" data-motion-key="node-O1"><circle cx="90" cy="58" r="23" /><text x="90" y="62">O1</text></g><g class="coding-trace-graph-node" data-motion-key="node-O2"><circle cx="42" cy="174" r="23" /><text x="42" y="178">O2</text></g><g class="coding-trace-graph-node" data-motion-key="node-O3"><circle cx="138" cy="174" r="23" /><text x="138" y="178">O3</text></g><g class="coding-trace-graph-node" data-motion-key="node-C1"><circle cx="390" cy="58" r="23" /><text x="390" y="62">C1</text></g></svg><div class="coding-trace-meta"><span><b>frontier</b>O1</span></div></div><div class="coding-trace-meta"><span><b>current</b>O1</span><span><b>copies</b>O1-&gt;C1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="copy-o2" hidden role="group" aria-label="Process original node O1"><div class="coding-trace-frame-heading"><span>Process original node O1</span><strong>Pop O1. Discover O2, create C2, enqueue O2, and append C2 to C1.neighbors.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><defs><marker id="coding-graph-arrow-8" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-graph-region" x="90" y="20">original</text><text class="coding-trace-graph-region" x="390" y="20">clone</text><g data-motion-key="edge-O1 &lt;-&gt; O2-0"><line class="coding-trace-edge-line" x1="80.05885338911621" y1="82.02443764296916" x2="53.08820198906269" y2="147.20351185976517" /></g><g data-motion-key="edge-O1 &lt;-&gt; O3-1"><line class="coding-trace-edge-line" x1="99.94114661088379" y1="82.02443764296916" x2="126.9117980109373" y2="147.20351185976517" /></g><g data-motion-key="edge-O2 &lt;-&gt; O3-2"><line class="coding-trace-edge-line" x1="68" y1="174" x2="109" y2="174" /></g><g data-motion-key="edge-C1 -&gt; C2-3"><line class="coding-trace-edge-line" x1="380.0588533891162" y1="82.02443764296916" x2="353.08820198906267" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-8)" /></g><g class="coding-trace-graph-node is-focus" data-motion-key="node-O1"><circle cx="90" cy="58" r="23" /><text x="90" y="62">O1</text></g><g class="coding-trace-graph-node" data-motion-key="node-O2"><circle cx="42" cy="174" r="23" /><text x="42" y="178">O2</text></g><g class="coding-trace-graph-node" data-motion-key="node-O3"><circle cx="138" cy="174" r="23" /><text x="138" y="178">O3</text></g><g class="coding-trace-graph-node" data-motion-key="node-C1"><circle cx="390" cy="58" r="23" /><text x="390" y="62">C1</text></g><g class="coding-trace-graph-node" data-motion-key="node-C2"><circle cx="342" cy="174" r="23" /><text x="342" y="178">C2</text></g></svg><div class="coding-trace-meta"><span><b>frontier</b>O2</span></div></div><div class="coding-trace-meta"><span><b>current</b>O1</span><span><b>copies</b>O1-&gt;C1, O2-&gt;C2</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="copy-o3" hidden role="group" aria-label="Finish O1 adjacency"><div class="coding-trace-frame-heading"><span>Finish O1 adjacency</span><strong>O1 next reaches unseen O3: create C3, enqueue O3, and append C3 to C1.neighbors. The queue is [O2,O3].</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><defs><marker id="coding-graph-arrow-9" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-graph-region" x="90" y="20">original</text><text class="coding-trace-graph-region" x="390" y="20">clone</text><g data-motion-key="edge-O1 &lt;-&gt; O2-0"><line class="coding-trace-edge-line" x1="80.05885338911621" y1="82.02443764296916" x2="53.08820198906269" y2="147.20351185976517" /></g><g data-motion-key="edge-O1 &lt;-&gt; O3-1"><line class="coding-trace-edge-line" x1="99.94114661088379" y1="82.02443764296916" x2="126.9117980109373" y2="147.20351185976517" /></g><g data-motion-key="edge-O2 &lt;-&gt; O3-2"><line class="coding-trace-edge-line" x1="68" y1="174" x2="109" y2="174" /></g><g data-motion-key="edge-C1 -&gt; C2-3"><line class="coding-trace-edge-line" x1="380.0588533891162" y1="82.02443764296916" x2="353.08820198906267" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-9)" /></g><g data-motion-key="edge-C1 -&gt; C3-4"><line class="coding-trace-edge-line" x1="399.9411466108838" y1="82.02443764296916" x2="426.91179801093733" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-9)" /></g><g class="coding-trace-graph-node is-focus" data-motion-key="node-O1"><circle cx="90" cy="58" r="23" /><text x="90" y="62">O1</text></g><g class="coding-trace-graph-node" data-motion-key="node-O2"><circle cx="42" cy="174" r="23" /><text x="42" y="178">O2</text></g><g class="coding-trace-graph-node" data-motion-key="node-O3"><circle cx="138" cy="174" r="23" /><text x="138" y="178">O3</text></g><g class="coding-trace-graph-node" data-motion-key="node-C1"><circle cx="390" cy="58" r="23" /><text x="390" y="62">C1</text></g><g class="coding-trace-graph-node" data-motion-key="node-C2"><circle cx="342" cy="174" r="23" /><text x="342" y="178">C2</text></g><g class="coding-trace-graph-node" data-motion-key="node-C3"><circle cx="438" cy="174" r="23" /><text x="438" y="178">C3</text></g></svg><div class="coding-trace-meta"><span><b>frontier</b>O2, O3</span></div></div><div class="coding-trace-meta"><span><b>current</b>O1</span><span><b>copies</b>O1-&gt;C1, O2-&gt;C2, O3-&gt;C3</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="connect-c2" hidden role="group" aria-label="Reuse copies while processing O2"><div class="coding-trace-frame-heading"><span>Reuse copies while processing O2</span><strong>Pop O2. Both neighbors are already mapped, so append existing C1 and C3 to C2.neighbors without creating or enqueueing another node.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><defs><marker id="coding-graph-arrow-10" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-graph-region" x="90" y="20">original</text><text class="coding-trace-graph-region" x="390" y="20">clone</text><g data-motion-key="edge-O1 &lt;-&gt; O2-0"><line class="coding-trace-edge-line" x1="80.05885338911621" y1="82.02443764296916" x2="53.08820198906269" y2="147.20351185976517" /></g><g data-motion-key="edge-O1 &lt;-&gt; O3-1"><line class="coding-trace-edge-line" x1="99.94114661088379" y1="82.02443764296916" x2="126.9117980109373" y2="147.20351185976517" /></g><g data-motion-key="edge-O2 &lt;-&gt; O3-2"><line class="coding-trace-edge-line" x1="68" y1="174" x2="109" y2="174" /></g><g data-motion-key="edge-C1 -&gt; C2-3"><line class="coding-trace-edge-line" x1="380.0588533891162" y1="82.02443764296916" x2="353.08820198906267" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-10)" /></g><g data-motion-key="edge-C1 -&gt; C3-4"><line class="coding-trace-edge-line" x1="399.9411466108838" y1="82.02443764296916" x2="426.91179801093733" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-10)" /></g><g data-motion-key="edge-C2 -&gt; C1-5"><line class="coding-trace-edge-line" x1="351.9411466108838" y1="149.97556235703084" x2="378.91179801093733" y2="84.79648814023483" marker-end="url(#coding-graph-arrow-10)" /></g><g data-motion-key="edge-C2 -&gt; C3-6"><line class="coding-trace-edge-line" x1="368" y1="174" x2="409" y2="174" marker-end="url(#coding-graph-arrow-10)" /></g><g class="coding-trace-graph-node" data-motion-key="node-O1"><circle cx="90" cy="58" r="23" /><text x="90" y="62">O1</text></g><g class="coding-trace-graph-node is-focus" data-motion-key="node-O2"><circle cx="42" cy="174" r="23" /><text x="42" y="178">O2</text></g><g class="coding-trace-graph-node" data-motion-key="node-O3"><circle cx="138" cy="174" r="23" /><text x="138" y="178">O3</text></g><g class="coding-trace-graph-node" data-motion-key="node-C1"><circle cx="390" cy="58" r="23" /><text x="390" y="62">C1</text></g><g class="coding-trace-graph-node" data-motion-key="node-C2"><circle cx="342" cy="174" r="23" /><text x="342" y="178">C2</text></g><g class="coding-trace-graph-node" data-motion-key="node-C3"><circle cx="438" cy="174" r="23" /><text x="438" y="178">C3</text></g></svg><div class="coding-trace-meta"><span><b>frontier</b>O3</span></div></div><div class="coding-trace-meta"><span><b>current</b>O2</span><span><b>copies</b>3 copies; no duplicates</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="connect-c3" hidden role="group" aria-label="Finish the copied adjacency"><div class="coding-trace-frame-heading"><span>Finish the copied adjacency</span><strong>Pop O3 and append existing C1 and C2 to C3.neighbors. The queue becomes empty after every original node is processed once.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><defs><marker id="coding-graph-arrow-11" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-graph-region" x="90" y="20">original</text><text class="coding-trace-graph-region" x="390" y="20">clone</text><g data-motion-key="edge-O1 &lt;-&gt; O2-0"><line class="coding-trace-edge-line" x1="80.05885338911621" y1="82.02443764296916" x2="53.08820198906269" y2="147.20351185976517" /></g><g data-motion-key="edge-O1 &lt;-&gt; O3-1"><line class="coding-trace-edge-line" x1="99.94114661088379" y1="82.02443764296916" x2="126.9117980109373" y2="147.20351185976517" /></g><g data-motion-key="edge-O2 &lt;-&gt; O3-2"><line class="coding-trace-edge-line" x1="68" y1="174" x2="109" y2="174" /></g><g data-motion-key="edge-C1 -&gt; C2-3"><line class="coding-trace-edge-line" x1="380.0588533891162" y1="82.02443764296916" x2="353.08820198906267" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-11)" /></g><g data-motion-key="edge-C1 -&gt; C3-4"><line class="coding-trace-edge-line" x1="399.9411466108838" y1="82.02443764296916" x2="426.91179801093733" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-11)" /></g><g data-motion-key="edge-C2 -&gt; C1-5"><line class="coding-trace-edge-line" x1="351.9411466108838" y1="149.97556235703084" x2="378.91179801093733" y2="84.79648814023483" marker-end="url(#coding-graph-arrow-11)" /></g><g data-motion-key="edge-C2 -&gt; C3-6"><line class="coding-trace-edge-line" x1="368" y1="174" x2="409" y2="174" marker-end="url(#coding-graph-arrow-11)" /></g><g data-motion-key="edge-C3 -&gt; C1-7"><line class="coding-trace-edge-line" x1="428.0588533891162" y1="149.97556235703084" x2="401.08820198906267" y2="84.79648814023483" marker-end="url(#coding-graph-arrow-11)" /></g><g data-motion-key="edge-C3 -&gt; C2-8"><line class="coding-trace-edge-line" x1="412" y1="174" x2="371" y2="174" marker-end="url(#coding-graph-arrow-11)" /></g><g class="coding-trace-graph-node" data-motion-key="node-O1"><circle cx="90" cy="58" r="23" /><text x="90" y="62">O1</text></g><g class="coding-trace-graph-node" data-motion-key="node-O2"><circle cx="42" cy="174" r="23" /><text x="42" y="178">O2</text></g><g class="coding-trace-graph-node is-focus" data-motion-key="node-O3"><circle cx="138" cy="174" r="23" /><text x="138" y="178">O3</text></g><g class="coding-trace-graph-node" data-motion-key="node-C1"><circle cx="390" cy="58" r="23" /><text x="390" y="62">C1</text></g><g class="coding-trace-graph-node" data-motion-key="node-C2"><circle cx="342" cy="174" r="23" /><text x="342" y="178">C2</text></g><g class="coding-trace-graph-node" data-motion-key="node-C3"><circle cx="438" cy="174" r="23" /><text x="438" y="178">C3</text></g></svg><div class="coding-trace-meta"><span><b>frontier</b></span></div></div><div class="coding-trace-meta"><span><b>current</b>O3</span><span><b>copies</b>O1-&gt;C1, O2-&gt;C2, O3-&gt;C3</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="return-deep-copy" hidden role="group" aria-label="Return the independent component"><div class="coding-trace-frame-heading"><span>Return the independent component</span><strong>Return C1. Copies C1,C2,C3 have the triangle adjacency of O1,O2,O3, but every copied edge points only to copied nodes.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><defs><marker id="coding-graph-arrow-12" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-graph-region" x="90" y="20">original</text><text class="coding-trace-graph-region" x="390" y="20">clone</text><g data-motion-key="edge-O1 &lt;-&gt; O2-0"><line class="coding-trace-edge-line" x1="80.05885338911621" y1="82.02443764296916" x2="53.08820198906269" y2="147.20351185976517" /></g><g data-motion-key="edge-O1 &lt;-&gt; O3-1"><line class="coding-trace-edge-line" x1="99.94114661088379" y1="82.02443764296916" x2="126.9117980109373" y2="147.20351185976517" /></g><g data-motion-key="edge-O2 &lt;-&gt; O3-2"><line class="coding-trace-edge-line" x1="68" y1="174" x2="109" y2="174" /></g><g data-motion-key="edge-C1 -&gt; C2-3"><line class="coding-trace-edge-line" x1="380.0588533891162" y1="82.02443764296916" x2="353.08820198906267" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-12)" /></g><g data-motion-key="edge-C1 -&gt; C3-4"><line class="coding-trace-edge-line" x1="399.9411466108838" y1="82.02443764296916" x2="426.91179801093733" y2="147.20351185976517" marker-end="url(#coding-graph-arrow-12)" /></g><g data-motion-key="edge-C2 -&gt; C1-5"><line class="coding-trace-edge-line" x1="351.9411466108838" y1="149.97556235703084" x2="378.91179801093733" y2="84.79648814023483" marker-end="url(#coding-graph-arrow-12)" /></g><g data-motion-key="edge-C2 -&gt; C3-6"><line class="coding-trace-edge-line" x1="368" y1="174" x2="409" y2="174" marker-end="url(#coding-graph-arrow-12)" /></g><g data-motion-key="edge-C3 -&gt; C1-7"><line class="coding-trace-edge-line" x1="428.0588533891162" y1="149.97556235703084" x2="401.08820198906267" y2="84.79648814023483" marker-end="url(#coding-graph-arrow-12)" /></g><g data-motion-key="edge-C3 -&gt; C2-8"><line class="coding-trace-edge-line" x1="412" y1="174" x2="371" y2="174" marker-end="url(#coding-graph-arrow-12)" /></g><g class="coding-trace-graph-node" data-motion-key="node-O1"><circle cx="90" cy="58" r="23" /><text x="90" y="62">O1</text></g><g class="coding-trace-graph-node" data-motion-key="node-O2"><circle cx="42" cy="174" r="23" /><text x="42" y="178">O2</text></g><g class="coding-trace-graph-node" data-motion-key="node-O3"><circle cx="138" cy="174" r="23" /><text x="138" y="178">O3</text></g><g class="coding-trace-graph-node" data-motion-key="node-C1"><circle cx="390" cy="58" r="23" /><text x="390" y="62">C1</text></g><g class="coding-trace-graph-node" data-motion-key="node-C2"><circle cx="342" cy="174" r="23" /><text x="342" y="178">C2</text></g><g class="coding-trace-graph-node" data-motion-key="node-C3"><circle cx="438" cy="174" r="23" /><text x="438" y="178">C3</text></g></svg></div><div class="coding-trace-meta"><span><b>current</b>return C1</span><span><b>copies</b>all clone edges stay inside C1,C2,C3</span><span><b>result</b>deep copy rooted at C1</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Create the first copy</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Process original node O1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Finish O1 adjacency</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Reuse copies while processing O2</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Finish the copied adjacency</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Return the independent component</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">A cyclic connected object must be deep-copied while preserving adjacency, so each original identity needs exactly one reusable copied identity before edges are wired.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Every key in copies maps to exactly one clone, and after an original node is processed, its clone has one copied neighbor entry for every original neighbor processed in order.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When copying any cyclic object graph, allocate and memoize an object before traversing its references, then resolve every copied reference through that memo.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For triangle O1-O2-O3, create C1 before traversal and enqueue O1. The map is {O1:C1}. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

Invariant: every discovered original already has exactly one allocated copy, although its outgoing adjacency may still be incomplete.
For each adjacency entry, append the mapped neighbor even when that neighbor was discovered earlier.

## Code

This reference supplies identity-hashable `GraphNode` objects.
It does not define value-based equality, so equal labels do not merge dictionary keys.

```python
from __future__ import annotations
from collections import deque

class GraphNode:
   def __init__(self, val: int = 0, neighbors: list[GraphNode] | None = None) -> None:
      self.val = val
      self.neighbors = neighbors or []


def clone_graph(node: GraphNode | None) -> GraphNode | None:
   if node is None:
      return None

   copies = {node: GraphNode(node.val)}
   queue = deque([node])

   while queue:
      current = queue.popleft()
      for neighbor in current.neighbors:
         if neighbor not in copies:
            copies[neighbor] = GraphNode(neighbor.val)
            queue.append(neighbor)
         copies[current].neighbors.append(copies[neighbor])
   return copies[node]
```

For reachable nodes and adjacency entries, time is $O(V+E)$.
Memo and queue use $O(V)$ auxiliary space.
Copied nodes and adjacency separately occupy $O(V+E)$ output space.
Repeated neighbor entries count separately in $E$.
Original nodes and lists remain unchanged.

### Failure Check

Memoizing after following edges can recurse forever on a cycle.
A visited Boolean alone cannot identify the copied target.
Skipping edges to visited nodes drops back edges, shared references, and repeated entries.
Keying by `val` incorrectly merges distinct equal-valued originals.

<!-- learning:core-end -->

## Change the Traversal

**New task:** Use recursive DFS while preserving the same copying contract.

**Picture and change:** Keep allocation before descent.
Every recursive revisit returns the already allocated copy.
Use the supplied `GraphNode` above.

```python
def clone_graph_dfs(node):
   copies = {}
   def visit(original):
      if original is None:
         return None
      if original not in copies:
         copies[original] = GraphNode(original.val)
         copies[original].neighbors = [visit(neighbor) for neighbor in original.neighbors]
      return copies[original]
   return visit(node)
```

**Check:** Give two distinct `7` nodes a back edge and two repeated references to the second node.
Their copies remain distinct, repeated references share one target, and the back edge closes inside the copy.
Time and output bounds remain unchanged; recursive depth can reach $V$ and exceed Python's limit.

**Use when:** DFS presentation is useful and recursive depth is acceptable.

## Change the Root Boundary

**New task:** Copy several roots, including disconnected or overlapping components.

**Picture and change:** Seed one shared memo and queue.
Pass a sequence, since `roots` is read twice.

```python
def clone_roots(roots):
   copies, queue = {}, deque()
   for root in roots:
      if root is not None and root not in copies:
         copies[root] = GraphNode(root.val)
         queue.append(root)
   while queue:
      original = queue.popleft()
      for neighbor in original.neighbors:
         if neighbor not in copies:
            copies[neighbor] = GraphNode(neighbor.val)
            queue.append(neighbor)
         copies[original].neighbors.append(copies[neighbor])
   return [None if root is None else copies[root] for root in roots]
```

**Check:** `[first, second, first, None]` returns the same copied `first` twice, an independent copied `second`, and `None`.
Shared reachable nodes are copied once across roots.
For $r$ roots, time is $O(r+V+E)$ and output space is $O(r+V+E)$; memo/queue remain $O(V)$.
Keep the reference's `deque` import.

**Use when:** Multiple entry points define the requested ownership boundary.

### Transfer Check

Why is the memo more than a visited set?

### Answer

It identifies the unique copied target needed for every reconstructed reference.
Discovery alone prevents repeated work but cannot rebuild the edges.
Compare [tree copying](/questions/invert-binary-tree/) when cycles and shared nodes are excluded by contract.
