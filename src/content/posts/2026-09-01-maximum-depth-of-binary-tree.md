---
title: "Return one subtree height after both children"
description: "Combine child heights for a longest path, then change execution depth or count every node."
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

Measure the longest root-to-leaf path in a binary tree by counting its nodes.

## Problem

**Input.** The root of a finite binary tree with `val`, `left`, and `right` fields.

**Output.** Return the greatest number of nodes on any path from the root to a leaf.

**Rules.** Count nodes, not edges. A leaf has depth one and an empty tree has depth zero.
Assume no shared nodes or cycles. Leave node values and child links unchanged.

**Example.** A root 3 has children 9 and 20; node 20 has children 15 and 7.
Return `3`, for either path `3 -> 20 -> 15` or `3 -> 20 -> 7`.
Its compact level-order notation is `[3, 9, 20, null, null, 15, 7]`.

## Picture

Each child sends one number upward, not its whole path.
Leaves `9`, `15`, and `7` each return `1`.
Node `20` combines `1` and `1` into `2`; root `3` combines `1` and `2` into `3`.

<!-- visual:maximum-depth-of-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="maximum-depth-of-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="maximum-depth-of-binary-tree-state-title">Add the current node to the larger returned child depth</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="maximum-depth-of-binary-tree" role="group" tabindex="0" aria-label="Maximum Depth of Binary Tree: Each recursive call returns the height of its subtree, so a parent needs only 1 + max(left height, right height)."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the number of nodes on the longest root-to-leaf path.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="call-root" role="group" aria-label="Start the root call"><div class="coding-trace-frame-heading"><span>Start the root call</span><strong>For tree [3,9,20,null,null,15,7], call max_depth(3). The stack contains [3], and neither child height is known yet.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-3-0"><b>3</b><small>call max_depth(3)</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[3]</span><span><b>knownReturns</b>none</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>call max_depth(3)</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>call stack</dt><dd><strong class="coding-state-after">[3]</strong></dd></div><div class="coding-state-change"><dt>known returns</dt><dd><strong class="coding-state-after">none</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="return-9" hidden role="group" aria-label="Resolve node 9 from null children"><div class="coding-trace-frame-heading"><span>Resolve node 9 from null children</span><strong>The left call reaches node 9. Both child calls receive None and return 0, so node 9 returns 1 + max(0,0) = 1.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-9-0"><b>9</b><small>return 1</small></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[3,9]</span><span><b>baseCases</b>None -&gt; 0 twice</span><span><b>arithmetic</b>1 + max(0,0) = 1</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>return 1</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>call max_depth(3)</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>call stack</dt><dd><span class="coding-state-before">[3]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[3,9]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="return-leaves" hidden role="group" aria-label="Resolve leaves 15 and 7"><div class="coding-trace-frame-heading"><span>Resolve leaves 15 and 7</span><strong>Under node 20, each leaf also receives two zero returns from None and returns 1.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><b>20</b></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-15-0"><b>15</b><small>return 1</small></span></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-7-0"><b>7</b><small>return 1</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[3,20]</span><span><b>returns</b>depth(15)=1; depth(7)=1</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>return 1</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">6</strong></dd></div><div class="coding-state-change"><dt>call stack</dt><dd><span class="coding-state-before">[3,9]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[3,20]</strong></dd></div><div class="coding-state-change"><dt>returns</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">depth(15)=1; depth(7)=1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="return-20" hidden role="group" aria-label="Combine at node 20"><div class="coding-trace-frame-heading"><span>Combine at node 20</span><strong>Node 20 receives left = 1 and right = 1, then returns 1 + max(1,1) = 2 to root 3.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-20-0"><b>20</b><small>return 2</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-15-0"><b>15</b><small>left 1</small></span></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-7-0"><b>7</b><small>right 1</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[3,20]</span><span><b>arithmetic</b>1 + max(1,1) = 2</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>return 2</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>left 1</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5</strong></dd></div><div class="coding-state-change"><dt>right 1</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">6</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="return-root" hidden role="group" aria-label="Combine at the root"><div class="coding-trace-frame-heading"><span>Combine at the root</span><strong>Root 3 receives left = 1 from node 9 and right = 2 from node 20, so it returns 1 + max(1,2) = 3.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-3-0"><b>3</b><small>return 3</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-9-0"><b>9</b><small>left 1</small></span></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-20-0"><b>20</b><small>right 2</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><b>15</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[]</span><span><b>arithmetic</b>1 + max(1,2) = 3</span><span><b>result</b>3</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>return 3</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>left 1</dt><dd><span class="coding-state-before">5</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>right 2</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start the root call</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Resolve node 9 from null children</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Resolve leaves 15 and 7</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Combine at node 20</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Combine at the root</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use this pattern when a tree property for a node can be computed only after both child subtrees return summaries, especially height, diameter contributions, or root-to-leaf aggregates.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Whenever max_depth(node) returns d, d is exactly the number of real nodes on the longest path from node to a leaf; max_depth(None) returns 0, making a leaf return 1.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Define the null identity first, ask each child for the smallest sufficient summary, and combine those summaries once; the same postorder fold supports subtree size, height, and path calculations.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For tree [3,9,20,null,null,15,7], call max_depth(3). The stack contains [3], and neither child height is known yet. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="maximum-depth-of-binary-tree-state"><strong>Check:</strong> Why does the root return three when its child heights are one and two?</span></figcaption></figure>

Every path through a real node uses that node, then at most one child path.
Therefore its node-count height is:

$$
H(\varnothing)=0,\qquad H(v)=1+\max(H(v.left),H(v.right)).
$$

The returned number is the exact height of that subtree.
Node values do not enter the recurrence.

## Code

```python
def max_depth(root: TreeNode | None) -> int:
   if root is None:
      return 0
   return 1 + max(max_depth(root.left), max_depth(root.right))
```

Each node is visited once: $O(n)$ time and $O(h)$ recursive stack space.
A chain has $h=n$, not $\log n$.
Python can raise `RecursionError` on a deep tree before the mathematical computation finishes.

### Failure Check

Counting edges changes the convention: the source tree has two edges on its longest path, but three nodes.
Using addition instead of `max` answers a different question: it returns `5` for this tree.

<!-- learning:core-end -->

## Change the Execution

**New task:** Trees can exceed Python's recursive call limit.

**Picture and change:** Carry `(node, depth)` work items explicitly.
`best` is the greatest depth among nodes already removed from the stack.

```python
def max_depth_iterative(root):
   stack = [] if root is None else [(root, 1)]
   best = 0
   while stack:
      node, depth = stack.pop()
      best = max(best, depth)
      if node.left is not None:
         stack.append((node.left, depth + 1))
      if node.right is not None:
         stack.append((node.right, depth + 1))
   return best
```

**Check:** The source tree still returns `3`; an empty tree returns `0`.
This uses $O(n)$ time and $O(h)$ auxiliary space on a binary tree.
It removes recursive calls, not memory needs or the no-cycle assumption.

**Use when:** Execution depth changes but the requested height does not.

## Change the Aggregate

**New task:** Count every node, not one longest path.

**Picture and change:** Both child contributions now matter, so replace `max` with addition.
The returned fact becomes subtree size.

```python
def tree_size(root):
   if root is None:
      return 0
   return 1 + tree_size(root.left) + tree_size(root.right)
```

**Check:** The source tree has size `5` and height `3`.
Empty and singleton trees have sizes `0` and `1`.
Time remains $O(n)$, with $O(h)$ recursive stack space and the same recursion limit.

**Use when:** All descendants contribute to the result.

### Transfer Check

Why does the parent not need the actual longest path?

### Answer

The output requests only a length.
The larger child height contains everything needed for that scalar result.
Returning the path itself would require a different summary.

Compare [level-order traversal](/questions/binary-tree-level-order-traversal/) when the task requires whole levels rather than one returned height.
