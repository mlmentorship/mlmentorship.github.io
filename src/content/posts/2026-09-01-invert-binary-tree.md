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

Mirror an entire binary tree in place while retaining the identities of its existing nodes.

## Problem

**Input.** The root of an owned finite binary tree, with `val`, `left`, and `right` fields on each node.

**Output.** Return the original root object after mirroring the complete tree.

**Rules.** Exchange left and right children at every node, not only at the root.
Reuse existing nodes and retain their values; do not create a separate copied tree.
Assume no cycles or shared nodes. An empty tree returns `None`.

**Example.** A root 4 with children 2 and 7 becomes a root 4 with children 7 and 2.
The full level-order tree `[4, 2, 7, 1, 3, 6, 9]` becomes `[4, 7, 2, 9, 6, 3, 1]`.

## Picture

Turn each branch into its mirror before attaching it to the opposite side.
Old right subtree `7` becomes `7(9,6)`.
Old left subtree `2` becomes `2(3,1)`.
Root `4` then connects left to `7` and right to `2`.

<!-- visual:invert-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="invert-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="invert-binary-tree-state-title">Swap returned subtrees, not the values stored in their nodes</p><div class="visual-argument" data-visual-argument="invert-binary-tree-state" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="Worked example: swap subtree positions without changing their stored values. Root 1 keeps value 1 while its returned left and right children exchange positions. Replacing values alone is not the recursive subtree operation."><title>Worked example: swap subtree positions without changing their stored values</title><desc>Root 1 keeps value 1 while its returned left and right children exchange positions. Replacing values alone is not the recursive subtree operation.</desc><text x="119.36666666666667" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="119.36666666666667" dy="0">root</tspan></text><text x="207.62" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="207.62" dy="0">left</tspan></text><text x="295.87333333333333" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="295.87333333333333" dy="0">right</tspan></text><text x="63.24000000000001" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="63.24000000000001" dy="0">before</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="75.24000000000001" y="36" width="83.25333333333333" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="116.86666666666667" y="74" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="116.86666666666667" dy="0">1</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="163.49333333333334" y="36" width="83.25333333333333" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="205.12" y="74" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="205.12" dy="0">2</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,2" x="251.74666666666667" y="36" width="83.25333333333333" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="293.37333333333333" y="74" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="293.37333333333333" dy="0">3</tspan></text></g><text x="63.24000000000001" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="63.24000000000001" dy="0">after</tspan></text><g data-proof-status="value"><rect data-proof-cell="1,0" x="75.24000000000001" y="100" width="83.25333333333333" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="116.86666666666667" y="138" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="116.86666666666667" dy="0">1</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="163.49333333333334" y="100" width="83.25333333333333" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="205.12" y="138" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="205.12" dy="0">3</tspan></text><path d="M168.49333333333334 153L241.74666666666667 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="value"><rect data-proof-cell="1,2" x="251.74666666666667" y="100" width="83.25333333333333" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="293.37333333333333" y="138" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="293.37333333333333" dy="0">2</tspan></text><path d="M256.74666666666667 153L330 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="invert-binary-tree" role="group" tabindex="0" aria-label="Return transformed child roots, then reconnect them locally: Postorder recursion returns inverted child subtrees, then the parent assigns old right to left and old left to right."><div class="coding-visual-example"><span>Input and goal</span><strong>Swap the left and right children at every node.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="call-root" role="group" aria-label="Start with the original links"><div class="coding-trace-frame-heading"><span>Start with the original links</span><strong>For tree [4,2,7,1,3,6,9], call invert_tree(4). The root currently links left to 2 and right to 7.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-4-0"><b>4</b><small>call invert(4)</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[4]</span><span><b>links</b>4.left=2; 4.right=7</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>node 4</dt><dd><strong class="coding-state-after">call invert(4)</strong></dd></div><div class="coding-state-change"><dt>call stack</dt><dd><strong class="coding-state-after">[4]</strong></dd></div><div class="coding-state-change"><dt>links</dt><dd><strong class="coding-state-after">4.left=2; 4.right=7</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="invert-right-subtree" hidden role="group" aria-label="Evaluate the old right subtree first"><div class="coding-trace-frame-heading"><span>Evaluate the old right subtree first</span><strong>The tuple right-hand side evaluates invert_tree(root.right) first. At node 7, invert 9 and 6; both leaves keep their values and null children.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li></ul></li><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-7-0"><b>7</b><small>invert old right</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-6-0"><b>6</b><small>returns 6</small></span></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-9-0"><b>9</b><small>returns 9</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[4,7]</span><span><b>evaluationOrder</b>invert(9), then invert(6)</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 7</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">invert old right</strong></dd></div><div class="coding-state-change"><dt>node 6</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">returns 6</strong></dd></div><div class="coding-state-change"><dt>node 9</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">returns 9</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="rewire-7" hidden role="group" aria-label="Rewire node 7"><div class="coding-trace-frame-heading"><span>Rewire node 7</span><strong>Assign node 7 left = inverted old right (9) and right = inverted old left (6). Stable nodes 9 and 6 visibly exchange child positions.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li></ul></li><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-7-0"><b>7</b><small>7: left=9, right=6</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>returnValue</b>inverted subtree rooted at 7</span><span><b>callStack</b>[4]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 7</dt><dd><span class="coding-state-before">invert old right</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">7: left=9, right=6</strong></dd></div><div class="coding-state-change"><dt>node 6</dt><dd><span class="coding-state-before">returns 6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>node 9</dt><dd><span class="coding-state-before">returns 9</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="rewire-2" hidden role="group" aria-label="Invert and rewire the old left subtree"><div class="coding-trace-frame-heading"><span>Invert and rewire the old left subtree</span><strong>Next evaluate invert_tree(2): invert old right 3, then old left 1, and assign node 2 left = 3 and right = 1.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-2-0"><b>2</b><small>2: left=3, right=1</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>evaluationOrder</b>invert(3), then invert(1)</span><span><b>callStack</b>[4]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 2</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2: left=3, right=1</strong></dd></div><div class="coding-state-change"><dt>node 7</dt><dd><span class="coding-state-before">7: left=9, right=6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>evaluation order</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">invert(3), then invert(1)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="rewire-root" hidden role="group" aria-label="Swap the root links"><div class="coding-trace-frame-heading"><span>Swap the root links</span><strong>Both recursive results are ready. Assign root 4 left = inverted old right subtree 7 and right = inverted old left subtree 2.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-4-0"><b>4</b><small>left=7, right=2</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-7-0"><b>7</b><small>moved from right</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><b>9</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li></ul></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-2-0"><b>2</b><small>moved from left</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[]</span><span><b>result</b>[4,7,2,9,6,3,1]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 4</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">left=7, right=2</strong></dd></div><div class="coding-state-change"><dt>node 7</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">moved from right</strong></dd></div><div class="coding-state-change"><dt>node 2</dt><dd><span class="coding-state-before">2: left=3, right=1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">moved from left</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start with the original links</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Evaluate the old right subtree first</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Rewire node 7</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Invert and rewire the old left subtree</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Swap the root links</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use this pattern when every node applies the same local child-pointer transformation and the parent needs transformed child roots returned from recursion.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">invert_tree(node) returns the same node object after every edge in its subtree is mirrored: its new left is the fully inverted old right and its new right is the fully inverted old left.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For recursive structural edits, first obtain transformed child roots, then reconnect them locally and return the current root; this transfers to pruning, flattening, and persistent tree rewrites.</dd></div></dl></div><div class="visual-derivation-caption">For tree [4,2,7,1,3,6,9], call invert_tree(4). The root currently links left to 2 and right to 7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="invert-binary-tree-state"><strong>Check:</strong> At the root, which returned subtree becomes the new left child, and did any node value change?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="invert-binary-tree-state">Root 1 keeps value 1 while its returned left and right children exchange positions. Replacing values alone is not the recursive subtree operation.</span></figcaption></figure>

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
