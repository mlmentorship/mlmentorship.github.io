---
title: "Freeze the current frontier before adding the next level"
description: "Capture one queue boundary, preserve depth order, then select or reduce each complete level."
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

Return a binary tree's values grouped by depth, with each level listed from left to right.

## Problem

**Input.** The root of a finite binary tree whose nodes have a value and optional left and right children.

**Output.** Return one list of values per depth, starting with the root's level.

**Rules.** Within each level, preserve the nodes' left-to-right order.
Missing children contribute no output value or placeholder.
Assume no shared nodes or cycles. Leave the tree unchanged; an empty tree returns `[]`.

**Example.** A root 3 has children 9 and 20; node 20 has children 15 and 7.
Return `[[3], [9, 20], [15, 7]]`, not one flattened list.
Its compact level-order input notation is `[3, 9, 20, null, null, 15, 7]`.

## Picture

Imagine a boundary in the queue between this level and the children being added for the next.
At each outer-loop start, the queue contains exactly one complete level.
Capture its size before appending anything.

| Frozen size | Values removed | Queue after those removals |
| --- | --- | --- |
| `1` | `3` | `[9,20]` |
| `2` | `9,20` | `[15,7]` |
| `2` | `15,7` | `[]` |

<!-- visual:binary-tree-level-order-traversal-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-tree-level-order-traversal-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="binary-tree-level-order-traversal-state-title">Freeze this level&#39;s queue length before adding the next level</p><div class="visual-argument" data-visual-argument="binary-tree-level-order-traversal-state" data-argument-kind="ledger" data-argument-composition="transform"><svg viewBox="0 0 360 394" role="img" aria-label="Worked example: freeze the current level before enqueueing its children. The level begins with A and B, so process exactly two nodes. Newly appended children belong to the next level."><title>Worked example: freeze the current level before enqueueing its children</title><desc>The level begins with A and B, so process exactly two nodes. Newly appended children belong to the next level.</desc><circle cx="23" cy="24" r="10" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="23" y="28" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">1</tspan></text><text x="44" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Queue at level start; size = 2</tspan></text><g data-proof-token="A" data-proof-row="0" data-proof-position="0" data-proof-box="44,60,142,48"><rect x="44" y="60" width="142" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="115" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="115" dy="0">A</tspan></text></g><g data-proof-token="B" data-proof-row="0" data-proof-position="1" data-proof-box="192,60,142,48"><rect x="192" y="60" width="142" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="263" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="263" dy="0">B</tspan></text></g><path d="M23 41L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="140" r="10" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="23" y="144" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">2</tspan></text><text x="44" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">After A enqueues its children</tspan></text><g data-proof-token="B" data-proof-row="1" data-proof-position="0" data-proof-box="44,176,92.66666666666667,48"><rect x="44" y="176" width="92.66666666666667" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="90.33333333333334" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="90.33333333333334" dy="0">B</tspan></text></g><g data-proof-token="A1" data-proof-row="1" data-proof-position="1" data-proof-box="142.66666666666669,176,92.66666666666667,48"><rect x="142.66666666666669" y="176" width="92.66666666666667" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="189.00000000000003" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="189.00000000000003" dy="0">A1</tspan></text></g><g data-proof-token="A2" data-proof-row="1" data-proof-position="2" data-proof-box="241.33333333333334,176,92.66666666666667,48"><rect x="241.33333333333334" y="176" width="92.66666666666667" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="287.6666666666667" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="287.6666666666667" dy="0">A2</tspan></text></g><path d="M23 157L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 228L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 228L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="256" r="10" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="23" y="260" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">3</tspan></text><text x="44" y="260" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Do not treat new length 3 as this</tspan><tspan x="44" dy="18">level</tspan></text><g data-proof-token="process B" data-proof-row="2" data-proof-position="0" data-proof-box="44,310,142,48"><rect x="44" y="310" width="142" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="115" y="337" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="115" dy="0">process B</tspan></text></g><g data-proof-token="then stop" data-proof-row="2" data-proof-position="1" data-proof-box="192,310,142,48"><rect x="192" y="310" width="142" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="263" y="337" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="263" dy="0">then stop</tspan></text></g></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="binary-tree-level-order-traversal" role="group" tabindex="0" aria-label="Freeze the current frontier before adding the next level: Snapshotting the queue length separates the current tree level from children appended for the next level."><div class="coding-visual-example"><span>Input and goal</span><strong>Return tree values one level at a time.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="queue-root" role="group" aria-label="Queue the root"><div class="coding-trace-frame-heading"><span>Queue the root</span><strong>For tree [3,9,20,null,null,15,7], initialize answer = [] and queue = [3]. The drawn edges show 3 -&gt; 9, 3 -&gt; 20, 20 -&gt; 15, and 20 -&gt; 7.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-3-0"><b>3</b><small>queued</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>queueState</b>[3]</span><span><b>answer</b>[]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>node 3</dt><dd><strong class="coding-state-after">queued</strong></dd></div><div class="coding-state-change"><dt>queue state</dt><dd><strong class="coding-state-after">[3]</strong></dd></div><div class="coding-state-change"><dt>answer</dt><dd><strong class="coding-state-after">[]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="finish-level-0" hidden role="group" aria-label="Freeze level size 1"><div class="coding-trace-frame-heading"><span>Freeze level size 1</span><strong>Read len(queue) = 1 before adding children. Pop 3 into level [3], then enqueue its left child 9 and right child 20.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-3-0"><b>3</b><small>level [3]</small></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-9-0"><b>9</b><small>next queue</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><b>20</b><small>next queue</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>1</span><span><b>queueState</b>[9,20]</span><span><b>answer</b>[[3]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 3</dt><dd><span class="coding-state-before">queued</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">level [3]</strong></dd></div><div class="coding-state-change"><dt>node 9</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">next queue</strong></dd></div><div class="coding-state-change"><dt>node 20</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">next queue</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="measure-level-1" hidden role="group" aria-label="Freeze level size 2"><div class="coding-trace-frame-heading"><span>Freeze level size 2</span><strong>At the next while iteration, len(queue) = 2. That fixed count means 9 and 20 belong together even though processing them will append children.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-9-0"><b>9</b><small>1 of 2</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><b>20</b><small>2 of 2</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>2</span><span><b>queueState</b>[9,20] before processing</span><span><b>level</b>[]</span><span><b>answer</b>[[3]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 9</dt><dd><span class="coding-state-before">next queue</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1 of 2</strong></dd></div><div class="coding-state-change"><dt>node 20</dt><dd><span class="coding-state-before">next queue</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2 of 2</strong></dd></div><div class="coding-state-change"><dt>node 3</dt><dd><span class="coding-state-before">level [3]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl><div class="coding-numeric-transition"><svg viewBox="0 0 360 158" role="img" aria-label="frozen size: 1 before, 2 at the next shown checkpoint." style="display:block;width:100%;max-width:34rem;margin:auto"><title>frozen size: 1 before, 2 at the next shown checkpoint.</title><desc>Each row has its own zero-inclusive numerical scale. Circles mark the preceding authored checkpoint and squares the current checkpoint. Connecting segments compare these shown values, not an invented path of execution.</desc><text x="16" y="20" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text-soft)">SHOWN CHECKPOINTS</text><text x="344" y="20" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text-soft)">circle: before / square: now</text><g data-numeric-transition="state:frozenSize" data-before="1" data-after="2" data-minimum="0" data-maximum="2"><text x="16" y="45" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">frozen size</text><text x="344" y="45" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">1 -&gt; 2</text><path d="M28 96H332" style="stroke:var(--c-rule);fill:none"></path><path d="M28 65V111" style="stroke:var(--c-text-soft);stroke-dasharray:3 3;fill:none"></path><path d="M180 69L332 96" style="stroke:var(--viz-focus-stroke);stroke-width:2;fill:none"></path><circle data-checkpoint-before cx="180" cy="69" r="5" style="fill:var(--c-bg);stroke:var(--c-text-soft);stroke-width:2"></circle><rect data-checkpoint-after x="327" y="91" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="28" y="128" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">0</text><text x="332" y="128" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">2</text></g></svg></div></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="finish-level-1" hidden role="group" aria-label="Process exactly two nodes"><div class="coding-trace-frame-heading"><span>Process exactly two nodes</span><strong>Pop 9, which has no children; pop 20, then enqueue 15 and 7. Append level [9,20], leaving queue [15,7] for the next iteration.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-9-0"><b>9</b><small>level [9,20]</small></span></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-20-0"><b>20</b><small>level [9,20]</small></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-15-0"><b>15</b><small>next queue</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-7-0"><b>7</b><small>next queue</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>queueState</b>[15,7]</span><span><b>answer</b>[[3],[9,20]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 9</dt><dd><span class="coding-state-before">1 of 2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">level [9,20]</strong></dd></div><div class="coding-state-change"><dt>node 20</dt><dd><span class="coding-state-before">2 of 2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">level [9,20]</strong></dd></div><div class="coding-state-change"><dt>node 15</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">next queue</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="finish-level-2" hidden role="group" aria-label="Process the leaf level"><div class="coding-trace-frame-heading"><span>Process the leaf level</span><strong>Freeze len(queue) = 2, pop 15 and 7, and append level [15,7]. Neither leaf adds a child, so the queue becomes empty.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-15-0"><b>15</b><small>level [15,7]</small></span></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-7-0"><b>7</b><small>level [15,7]</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>2</span><span><b>queueState</b>[]</span><span><b>answer</b>[[3],[9,20],[15,7]]</span><span><b>result</b>[[3],[9,20],[15,7]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 15</dt><dd><span class="coding-state-before">next queue</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">level [15,7]</strong></dd></div><div class="coding-state-change"><dt>node 7</dt><dd><span class="coding-state-before">next queue</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">level [15,7]</strong></dd></div><div class="coding-state-change"><dt>node 9</dt><dd><span class="coding-state-before">level [9,20]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Queue the root</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Freeze level size 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Freeze level size 2</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Process exactly two nodes</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Process the leaf level</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use this BFS form when a tree answer is grouped by depth, processed left-to-right by level, or must compute one aggregate per level rather than one flat visitation order.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">At each while-loop start, the queue contains exactly one complete next level in left-to-right order. Iterating the captured length consumes only that level while appending its children for the following level.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Capture the frontier size before expanding it whenever output or timing is grouped by BFS depth; this transfers to right-side view, level averages, shortest unweighted paths, and wave simulations.</dd></div></dl></div><div class="visual-derivation-caption">For tree [3,9,20,null,null,15,7], initialize answer = [] and queue = [3]. The drawn edges show 3 -&gt; 9, 3 -&gt; 20, 20 -&gt; 15, and 20 -&gt; 7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="binary-tree-level-order-traversal-state"><strong>Check:</strong> Why do 15 and 7 wait for the next level even though they are already in the queue?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="binary-tree-level-order-traversal-state">The level begins with A and B, so process exactly two nodes. Newly appended children belong to the next level.</span></figcaption></figure>

During a level, the queue can mix unprocessed current nodes with already discovered next-level nodes.
The captured count prevents consuming those children too early.
Appending left before right preserves the requested order.

## Code

This complete reference includes its node definition.
`range(len(queue))` evaluates the length once, before the loop body changes the queue.

```python
from __future__ import annotations
from collections import deque
from dataclasses import dataclass

@dataclass(eq=False, slots=True)
class TreeNode:
   val: int
   left: TreeNode | None = None
   right: TreeNode | None = None

def level_order(root: TreeNode | None) -> list[list[int]]:
   if root is None:
      return []

   answer: list[list[int]] = []
   queue = deque([root])

   while queue:
      level = []
      for _ in range(len(queue)):
         node = queue.popleft()
         level.append(node.val)
         if node.left:
            queue.append(node.left)
         if node.right:
            queue.append(node.right)
      answer.append(level)
   return answer
```

Each node is queued and removed once: $O(n)$ time.
The queue uses $O(w)$ auxiliary space, where $w$ is the widest level.
It may hold parts of two adjacent levels, still within a constant multiple of $w$.
The returned lists separately occupy $O(n)$ space.

### Failure Check

Consuming until the queue empties inside one level also consumes newly appended children.
The source tree then collapses into `[3,9,20,15,7]`, losing the depth partition.
Graph inputs need a separate discovered-state contract.

<!-- learning:core-end -->

## Select One Value

**New task:** Return the visible right-side value at each depth.

**Picture and change:** Keep the same depth partition, then select the final left-to-right value.
Place this wrapper after the reference above.

```python
def right_side_values(root):
   return [level[-1] for level in level_order(root)]
```

**Check:** The source returns `[3,20,7]`; an empty tree returns `[]`.
The wrapper still materializes all levels temporarily: $O(n)$ peak storage, not just its $O(h)$ returned values.

**Use when:** One selected value per level is enough.

## Reduce Each Level

**New task:** Return one sum per depth.

**Picture and change:** Keep the partition and add each level's values.

```python
def level_sums(root):
   return [sum(level) for level in level_order(root)]
```

**Check:** The source returns `[3,29,22]`.
Negative and zero values also contribute normally.
Both wrappers take $O(n)$ time and leave input links unchanged.
Both retain the reference's $O(n)$ temporary level materialization.
A fused version could accumulate within each frozen frontier instead.

**Use when:** Only an aggregate per level is required.

### Transfer Check

Why not replace the FIFO queue with a stack while keeping this reasoning?

### Answer

A stack can visit newly added descendants before remaining siblings.
It does not preserve this complete next-depth frontier without additional state.
Compare [subtree heights](/questions/maximum-depth-of-binary-tree/) for a different traversal contract.
