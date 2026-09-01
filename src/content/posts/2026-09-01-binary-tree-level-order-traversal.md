---
title: "Binary Tree Level Order Traversal"
description: "Return tree values one level at a time."
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

> Return tree values one level at a time.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:binary-tree-level-order-traversal-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-tree-level-order-traversal-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="binary-tree-level-order-traversal-state-title">Binary Tree Level Order Traversal: Snapshotting the queue length separates the current tree level from children appended for the next level.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="binary-tree-level-order-traversal" role="group" tabindex="0" aria-label="Binary Tree Level Order Traversal: Snapshotting the queue length separates the current tree level from children appended for the next level."><div class="coding-visual-example"><span>Input and goal</span><strong>Return tree values one level at a time.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="queue-root" role="group" aria-label="Queue the root"><div class="coding-trace-frame-heading"><span>Queue the root</span><strong>For tree [3,9,20,null,null,15,7], initialize answer = [] and queue = [3]. The drawn edges show 3 -&gt; 9, 3 -&gt; 20, 20 -&gt; 15, and 20 -&gt; 7.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-3-0"><b>3</b><small>queued</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>queueState</b>[3]</span><span><b>answer</b>[]</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="finish-level-0" hidden role="group" aria-label="Freeze level size 1"><div class="coding-trace-frame-heading"><span>Freeze level size 1</span><strong>Read len(queue) = 1 before adding children. Pop 3 into level [3], then enqueue its left child 9 and right child 20.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-3-0"><b>3</b><small>level [3]</small></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-9-0"><b>9</b><small>next queue</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><b>20</b><small>next queue</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>1</span><span><b>queueState</b>[9,20]</span><span><b>answer</b>[[3]]</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="measure-level-1" hidden role="group" aria-label="Freeze level size 2"><div class="coding-trace-frame-heading"><span>Freeze level size 2</span><strong>At the next while iteration, len(queue) = 2. That fixed count means 9 and 20 belong together even though processing them will append children.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-9-0"><b>9</b><small>1 of 2</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><b>20</b><small>2 of 2</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>2</span><span><b>queueState</b>[9,20] before processing</span><span><b>level</b>[]</span><span><b>answer</b>[[3]]</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="finish-level-1" hidden role="group" aria-label="Process exactly two nodes"><div class="coding-trace-frame-heading"><span>Process exactly two nodes</span><strong>Pop 9, which has no children; pop 20, then enqueue 15 and 7. Append level [9,20], leaving queue [15,7] for the next iteration.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-9-0"><b>9</b><small>level [9,20]</small></span></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-20-0"><b>20</b><small>level [9,20]</small></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-15-0"><b>15</b><small>next queue</small></span></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-7-0"><b>7</b><small>next queue</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>queueState</b>[15,7]</span><span><b>answer</b>[[3],[9,20]]</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="finish-level-2" hidden role="group" aria-label="Process the leaf level"><div class="coding-trace-frame-heading"><span>Process the leaf level</span><strong>Freeze len(queue) = 2, pop 15 and 7, and append level [15,7]. Neither leaf adds a child, so the queue becomes empty.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-15-0"><b>15</b><small>level [15,7]</small></span></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-7-0"><b>7</b><small>level [15,7]</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>frozenSize</b>2</span><span><b>queueState</b>[]</span><span><b>answer</b>[[3],[9,20],[15,7]]</span><span><b>result</b>[[3],[9,20],[15,7]]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Queue the root</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Freeze level size 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Freeze level size 2</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Process exactly two nodes</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Process the leaf level</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use this BFS form when a tree answer is grouped by depth, processed left-to-right by level, or must compute one aggregate per level rather than one flat visitation order.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">At each while-loop start, the queue contains exactly one complete next level in left-to-right order. Iterating the captured length consumes only that level while appending its children for the following level.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Capture the frontier size before expanding it whenever output or timing is grouped by BFS depth; this transfers to right-side view, level averages, shortest unweighted paths, and wave simulations.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For tree [3,9,20,null,null,15,7], initialize answer = [] and queue = [3]. The drawn edges show 3 -&gt; 9, 3 -&gt; 20, 20 -&gt; 15, and 20 -&gt; 7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** BFS.

**Simple idea:** The queue contains the next level. Read its current size before adding any
children. That size tells you how many nodes belong to this level.

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

**Cost:** $O(n)$ time and $O(w)$ space, where $w$ is the widest level.
