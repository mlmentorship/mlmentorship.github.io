---
title: "Return transformed child roots, then reconnect them locally"
description: "Mirror every subtree while preserving node identities, then change execution or return a separate copy."
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

Mirror both child subtrees, attach their roots on opposite sides, and return the same current node.

## Problem

Mirror `[4,2,7,1,3,6,9]` into `[4,7,2,9,6,3,1]`.
Swap left and right at **every** node, not just the root.
Reuse the existing nodes and return the original root object.
An empty tree returns `None`.

Assume an owned, finite tree without cycles or shared nodes.
The platform supplies `TreeNode` with `val`, `left`, and `right` fields.

## Picture

Turn each branch into its mirror before attaching it to the opposite side.
Old right subtree `7` becomes `7(9,6)`.
Old left subtree `2` becomes `2(3,1)`.
Root `4` then connects left to `7` and right to `2`.

<!-- visual:invert-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="invert-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="invert-binary-tree-state-title">Invert Binary Tree: Postorder recursion returns inverted child subtrees, then the parent assigns old right to left and old left to right.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="invert-binary-tree" role="group" tabindex="0" aria-label="Invert Binary Tree: Postorder recursion returns inverted child subtrees, then the parent assigns old right to left and old left to right."><div class="coding-visual-example"><span>Input and goal</span><strong>Swap the left and right children at every node.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="call-root" role="group" aria-label="Start with the original links"><div class="coding-trace-frame-heading"><span>Start with the original links</span><strong>For tree [4,2,7,1,3,6,9], call invert_tree(4). The root currently links left to 2 and right to 7.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-4-0"><b>4</b><small>call invert(4)</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[4]</span><span><b>links</b>4.left=2; 4.right=7</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="invert-right-subtree" hidden role="group" aria-label="Evaluate the old right subtree first"><div class="coding-trace-frame-heading"><span>Evaluate the old right subtree first</span><strong>The tuple right-hand side evaluates invert_tree(root.right) first. At node 7, invert 9 and 6; both leaves keep their values and null children.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li></ul></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-7-0"><b>7</b><small>invert old right</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-6-0"><b>6</b><small>returns 6</small></span></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-9-0"><b>9</b><small>returns 9</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[4,7]</span><span><b>evaluationOrder</b>invert(9), then invert(6)</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="rewire-7" hidden role="group" aria-label="Rewire node 7"><div class="coding-trace-frame-heading"><span>Rewire node 7</span><strong>Assign node 7 left = inverted old right (9) and right = inverted old left (6). Stable nodes 9 and 6 visibly exchange child positions.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li></ul></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-7-0"><b>7</b><small>7: left=9, right=6</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>returnValue</b>inverted subtree rooted at 7</span><span><b>callStack</b>[4]</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="rewire-2" hidden role="group" aria-label="Invert and rewire the old left subtree"><div class="coding-trace-frame-heading"><span>Invert and rewire the old left subtree</span><strong>Next evaluate invert_tree(2): invert old right 3, then old left 1, and assign node 2 left = 3 and right = 1.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-2-0"><b>2</b><small>2: left=3, right=1</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>evaluationOrder</b>invert(3), then invert(1)</span><span><b>callStack</b>[4]</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="rewire-root" hidden role="group" aria-label="Swap the root links"><div class="coding-trace-frame-heading"><span>Swap the root links</span><strong>Both recursive results are ready. Assign root 4 left = inverted old right subtree 7 and right = inverted old left subtree 2.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-4-0"><b>4</b><small>left=7, right=2</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-7-0"><b>7</b><small>moved from right</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li></ul></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-2-0"><b>2</b><small>moved from left</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[]</span><span><b>result</b>[4,7,2,9,6,3,1]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start with the original links</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Evaluate the old right subtree first</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Rewire node 7</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Invert and rewire the old left subtree</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Swap the root links</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use this pattern when every node applies the same local child-pointer transformation and the parent needs transformed child roots returned from recursion.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">invert_tree(node) returns the same node object after every edge in its subtree is mirrored: its new left is the fully inverted old right and its new right is the fully inverted old left.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For recursive structural edits, first obtain transformed child roots, then reconnect them locally and return the current root; this transfers to pruning, flattening, and persistent tree rewrites.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For tree [4,2,7,1,3,6,9], call invert_tree(4). The root currently links left to 2 and right to 7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

The recursive return contract is: this same node now roots the complete mirror of its former subtree.
Values and node identities remain unchanged; child links change.

## Code

```python
def invert_tree(root: TreeNode | None) -> TreeNode | None:
   if root is None:
      return None
   root.left, root.right = invert_tree(root.right), invert_tree(root.left)
   return root
```

Python evaluates the old right recursion first, then the old left recursion.
Both results are ready before either assignment target changes.
This preserves both old child references during evaluation.

Each node is visited once: $O(n)$ time and $O(h)$ recursive stack space.
Height $h$ can equal $n$; Python's recursion limit still applies.

### Failure Check

Swapping only root children produces `[4,7,2,6,9,1,3]`, not the full mirror.
Assigning `root.left = root.right` before preserving old left loses access to subtree `2` through this root.

<!-- learning:core-end -->

## Change the Execution

**New task:** Keep in-place mutation without recursive calls.

**Picture and change:** Pop one node, swap its links, then schedule its real children.
Traversal order changes; the final mirror and node identities do not.

```python
def invert_iterative(root):
   stack = [] if root is None else [root]
   while stack:
      node = stack.pop()
      node.left, node.right = node.right, node.left
      if node.left is not None:
         stack.append(node.left)
      if node.right is not None:
         stack.append(node.right)
   return root
```

**Check:** The source tree reaches the same mirror.
Inverting twice restores every original link and value.
Time is $O(n)$, with $O(h)$ auxiliary space on a binary tree.

**Use when:** Mutation is permitted but recursive depth is unsuitable.

## Change the Ownership

**New task:** Leave the original tree unchanged.

**Picture and change:** Allocate one new node for each old node.
Assume the compatible constructor `TreeNode(val, left=None, right=None)`.

```python
def inverted_copy(root):
   if root is None:
      return None
   return TreeNode(root.val, inverted_copy(root.right), inverted_copy(root.left))
```

**Check:** Require mirrored values and links, unchanged original links, and disjoint original/copy node identities.
This takes $O(n)$ time, $O(n)$ new-node storage, and $O(h)$ recursive stack space.
It is not in-place or safe for arbitrary cyclic graphs.

**Use when:** Ownership forbids changing the input.

### Transfer Check

Why can the original assignment safely use both old child links?

### Answer

Both right-hand expressions finish before either left-hand target is assigned.
That ordering prevents overwriting the old left reference before its recursive call.
For shared or cyclic structures, establish an aliasing policy instead; compare [graph cloning](/questions/clone-graph/).
