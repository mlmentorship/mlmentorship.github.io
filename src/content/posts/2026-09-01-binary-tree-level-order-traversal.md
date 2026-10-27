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
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-tree-level-order-traversal-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="binary-tree-level-order-traversal-state-title">Binary Tree Level Order Traversal: Snapshotting the queue length separates the current tree level from children appended for the next level.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="binary-tree-level-order-traversal" role="group" tabindex="0" aria-label="Binary Tree Level Order Traversal: Snapshotting the queue length separates the current tree level from children appended for the next level."><div class="coding-visual-example"><span>Input and goal</span><strong>Return tree values one level at a time.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="queue-root" role="group" aria-label="Queue the root"><div class="coding-trace-frame-heading"><span>Queue the root</span><strong>For tree [3,9,20,null,null,15,7], initialize answer = [] and queue = [3]. The drawn edges show 3 -&gt; 9, 3 -&gt; 20, 20 -&gt; 15, and 20 -&gt; 7.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-3-0"><b>3</b><small>queued</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>queueState</b>[3]</span><span><b>answer</b>[]</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="finish-level-0" hidden role="group" aria-label="Freeze level size 1"><div class="coding-trace-frame-heading"><span>Freeze level size 1</span><strong>Read len(queue) = 1 before adding children. Pop 3 into level [3], then enqueue its left child 9 and right child 20.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-3-0"><b>3</b><small>level [3]</small></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-9-0"><b>9</b><small>next queue</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><b>20</b><small>next queue</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>1</span><span><b>queueState</b>[9,20]</span><span><b>answer</b>[[3]]</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="measure-level-1" hidden role="group" aria-label="Freeze level size 2"><div class="coding-trace-frame-heading"><span>Freeze level size 2</span><strong>At the next while iteration, len(queue) = 2. That fixed count means 9 and 20 belong together even though processing them will append children.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-9-0"><b>9</b><small>1 of 2</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><b>20</b><small>2 of 2</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>2</span><span><b>queueState</b>[9,20] before processing</span><span><b>level</b>[]</span><span><b>answer</b>[[3]]</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="finish-level-1" hidden role="group" aria-label="Process exactly two nodes"><div class="coding-trace-frame-heading"><span>Process exactly two nodes</span><strong>Pop 9, which has no children; pop 20, then enqueue 15 and 7. Append level [9,20], leaving queue [15,7] for the next iteration.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-9-0"><b>9</b><small>level [9,20]</small></span></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-20-0"><b>20</b><small>level [9,20]</small></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-15-0"><b>15</b><small>next queue</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-7-0"><b>7</b><small>next queue</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>queueState</b>[15,7]</span><span><b>answer</b>[[3],[9,20]]</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="finish-level-2" hidden role="group" aria-label="Process the leaf level"><div class="coding-trace-frame-heading"><span>Process the leaf level</span><strong>Freeze len(queue) = 2, pop 15 and 7, and append level [15,7]. Neither leaf adds a child, so the queue becomes empty.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-15-0"><b>15</b><small>level [15,7]</small></span></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-7-0"><b>7</b><small>level [15,7]</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>2</span><span><b>queueState</b>[]</span><span><b>answer</b>[[3],[9,20],[15,7]]</span><span><b>result</b>[[3],[9,20],[15,7]]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Queue the root</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Freeze level size 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Freeze level size 2</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Process exactly two nodes</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Process the leaf level</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use this BFS form when a tree answer is grouped by depth, processed left-to-right by level, or must compute one aggregate per level rather than one flat visitation order.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">At each while-loop start, the queue contains exactly one complete next level in left-to-right order. Iterating the captured length consumes only that level while appending its children for the following level.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Capture the frontier size before expanding it whenever output or timing is grouped by BFS depth; this transfers to right-side view, level averages, shortest unweighted paths, and wave simulations.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For tree [3,9,20,null,null,15,7], initialize answer = [] and queue = [3]. The drawn edges show 3 -&gt; 9, 3 -&gt; 20, 20 -&gt; 15, and 20 -&gt; 7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
