---
title: "Balanced Binary Tree"
description: "Return a certified subtree height or failure, propagate failure before combining, and distinguish local height from whole-tree balance."
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

Determine whether every node in a binary tree has subtrees of nearly equal height.

## Problem

**Input.** The root of a finite binary tree, with `left` and `right` child references or `None`.

**Output.** Return `True` exactly when every node's left and right subtree heights differ by at most one.

**Rules.** Height counts nodes: an empty subtree has height zero and a leaf has height one.
The condition must hold throughout the tree, not only at the root.
Assume no cycles or shared children. Do not change any nodes; an empty tree is balanced.

**Example.** A root with two leaf children returns `True`.
A three-node chain containing only left children returns `False`; its root's subtree heights are two and zero.

## Picture

Follow the left chain upward. Valid heights stop at node `2`; the root then skips its right subtree.

| Node | Left result | Right result | Returned result |
| --- | --- | --- | --- |
| 4 | 0 | 0 | 1 |
| 3 | 1 | 0 | 2 |
| 2 | 2 | 0 | -1: unbalanced |
| 1 | -1 | Not called | -1: propagate failure |

<!-- visual:balanced-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="balanced-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="balanced-binary-tree-state-title">An unbalanced child can end the check before further searching</p><div class="visual-argument" data-visual-argument="balanced-binary-tree-state" data-argument-kind="ledger" data-argument-composition="transform"><svg viewBox="0 0 360 260" role="img" aria-label="Worked example: a failed child cannot be repaired by a balanced parent. A child returns the unbalanced sentinel, not a usable height. Propagate that failure rather than averaging or comparing it as a valid depth."><title>Worked example: a failed child cannot be repaired by a balanced parent</title><desc>A child returns the unbalanced sentinel, not a usable height. Propagate that failure rather than averaging or comparing it as a valid depth.</desc><circle cx="23" cy="24" r="10" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="23" y="28" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">1</tspan></text><text x="44" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Returned child results</tspan></text><g data-proof-token="left: FAIL" data-proof-row="0" data-proof-position="0" data-proof-box="44,60,142,48"><rect x="44" y="60" width="142" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="115" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="115" dy="0">left: FAIL</tspan></text></g><g data-proof-token="right: 2" data-proof-row="0" data-proof-position="1" data-proof-box="192,60,142,48"><rect x="192" y="60" width="142" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="263" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="263" dy="0">right: 2</tspan></text></g><path d="M23 41L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="140" r="10" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="23" y="144" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">2</tspan></text><text x="44" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Parent result</tspan></text><g data-proof-token="FAIL" data-proof-row="1" data-proof-position="0" data-proof-box="44,176,290,48"><rect x="44" y="176" width="290" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect><text x="189" y="205.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="189" dy="0">FAIL</tspan></text></g></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="balanced-binary-tree" role="group" tabindex="0" aria-label="Balanced Binary Tree: A postorder height function returns a nonnegative height for a balanced subtree and -1 as a failure sentinel that ancestors propagate immediately."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether the child heights at every node differ by at most one.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="descend-left-chain" role="group" aria-label="Start bottom-up height evaluation"><div class="coding-trace-frame-heading"><span>Start bottom-up height evaluation</span><strong>Call height(1), then descend through 2 and 3 to leaf 4. No right sibling has been evaluated.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-1-0"><b>1</b><small>stack start</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-2-0"><b>2</b><small>stack</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-3-0"><b>3</b><small>stack</small></span><ul><li><span class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-4-0"><b>4</b><small>current</small></span></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><b>5</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>callStack</b>[1,2,3,4]</span><span><b>untouchedRightSubtree</b>5 with children 6,7</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>node 1</dt><dd><strong class="coding-state-after">stack start</strong></dd></div><div class="coding-state-change"><dt>node 2</dt><dd><strong class="coding-state-after">stack</strong></dd></div><div class="coding-state-change"><dt>node 3</dt><dd><strong class="coding-state-after">stack</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="return-leaf-4" hidden role="group" aria-label="Return height 1 from leaf 4"><div class="coding-trace-frame-heading"><span>Return height 1 from leaf 4</span><strong>Node 4 gets left = 0 and right = 0 from its None children. The difference is 0, so return 1 + max(0,0) = 1.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-4-0"><b>4</b><small>return height 1</small></span></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><b>5</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>arithmetic</b>|0-0| = 0 &lt;= 1; 1 + max(0,0) = 1</span><span><b>callStack</b>[1,2,3]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 4</dt><dd><span class="coding-state-before">current</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">return height 1</strong></dd></div><div class="coding-state-change"><dt>node 1</dt><dd><span class="coding-state-before">stack start</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>node 2</dt><dd><span class="coding-state-before">stack</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="return-node-3" hidden role="group" aria-label="Return height 2 from node 3"><div class="coding-trace-frame-heading"><span>Return height 2 from node 3</span><strong>Node 3 receives left = 1 and right = 0. Since |1-0| = 1, it is balanced and returns 1 + max(1,0) = 2.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-3-0"><b>3</b><small>return height 2</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-4-0"><b>4</b><small>left height 1</small></span></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><b>5</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>arithmetic</b>|1-0| = 1; 1 + max(1,0) = 2</span><span><b>callStack</b>[1,2]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 3</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">return height 2</strong></dd></div><div class="coding-state-change"><dt>node 4</dt><dd><span class="coding-state-before">return height 1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">left height 1</strong></dd></div><div class="coding-state-change"><dt>arithmetic</dt><dd><span class="coding-state-before">|0-0| = 0 &lt;= 1; 1 + max(0,0) = 1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">|1-0| = 1; 1 + max(1,0) = 2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="emit-sentinel" hidden role="group" aria-label="Emit the failure sentinel at node 2"><div class="coding-trace-frame-heading"><span>Emit the failure sentinel at node 2</span><strong>Node 2 receives left = 2 and right = 0. Because |2-0| = 2 &gt; 1, it returns -1 instead of a height.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><b>1</b></span><ul><li><span class="coding-trace-tree-node trace-tone-warning" data-motion-key="tree-node-2-0"><b>2</b><small>return -1</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-3-0"><b>3</b><small>left height 2</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><b>5</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>arithmetic</b>|2-0| = 2 &gt; 1</span><span><b>sentinel</b>-1 means unbalanced</span><span><b>callStack</b>[1]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 2</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">return -1</strong></dd></div><div class="coding-state-change"><dt>node 3</dt><dd><span class="coding-state-before">return height 2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">left height 2</strong></dd></div><div class="coding-state-change"><dt>node 4</dt><dd><span class="coding-state-before">left height 1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="propagate-sentinel" hidden role="group" aria-label="Propagate -1 and skip the right subtree"><div class="coding-trace-frame-heading"><span>Propagate -1 and skip the right subtree</span><strong>Root 1 receives left = -1. The `if left &lt; 0` branch returns -1 immediately, so nodes 5, 6, and 7 are never visited.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-warning" data-motion-key="tree-node-1-0"><b>1</b><small>propagate -1</small></span><ul><li><span class="coding-trace-tree-node trace-tone-warning" data-motion-key="tree-node-2-0"><b>2</b><small>left returned -1</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-5-0"><b>5</b><small>skipped</small></span><ul><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-6-0"><b>6</b><small>skipped</small></span></li><li><span class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-7-0"><b>7</b><small>skipped</small></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>branch</b>left &lt; 0 -&gt; return -1</span><span><b>visited</b>1,2,3,4 only</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 1</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">propagate -1</strong></dd></div><div class="coding-state-change"><dt>node 2</dt><dd><span class="coding-state-before">return -1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">left returned -1</strong></dd></div><div class="coding-state-change"><dt>node 5</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">skipped</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="return-false" hidden role="group" aria-label="Convert the sentinel to false"><div class="coding-trace-frame-heading"><span>Convert the sentinel to false</span><strong>The outer function checks height(root) &gt;= 0. Here -1 &gt;= 0 is false, so the tree is not height-balanced.</strong></div><div class="coding-trace-tree" role="img" aria-label="Binary tree with parent-child edges and call state"><ul class="coding-trace-tree-semantic"><li><span class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-1-0"><b>1</b><small>height(root) = -1</small></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><b>2</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><b>3</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><b>4</b></span></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li class="is-placeholder" aria-hidden="true"><span></span></li></ul></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><b>5</b></span><ul><li><span class="coding-trace-tree-node" data-motion-key="tree-node-6-0"><b>6</b></span></li><li><span class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><b>7</b></span></li></ul></li></ul></li></ul></div><div class="coding-trace-meta"><span><b>arithmetic</b>-1 &gt;= 0 -&gt; false</span><span><b>result</b>false</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node 1</dt><dd><span class="coding-state-before">propagate -1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">height(root) = -1</strong></dd></div><div class="coding-state-change"><dt>node 2</dt><dd><span class="coding-state-before">left returned -1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>node 5</dt><dd><span class="coding-state-before">skipped</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start bottom-up height evaluation</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Return height 1 from leaf 4</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return height 2 from node 3</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Emit the failure sentinel at node 2</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Propagate -1 and skip the right subtree</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Convert the sentinel to false</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use a sentinel summary when a parent needs a normal child aggregate but any descendant failure should abort remaining work, as in balanced-height checks or invalid-subtree detection.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">height(node) returns the exact nonnegative subtree height if every node below is balanced; otherwise it returns -1. Thus ancestors can distinguish valid data from failure without a second traversal.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Fuse a subtree summary with validation by reserving an impossible summary value for failure, then check it before doing more recursion; this generalizes to BST validation and parse-tree error propagation.</dd></div></dl></div><div class="visual-derivation-caption">Call height(1), then descend through 2 and 3 to leaf 4. No right sibling has been evaluated. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="balanced-binary-tree-state"><strong>Check:</strong> Why does the root return without visiting nodes 5, 6 and 7?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="balanced-binary-tree-state">A child returns the unbalanced sentinel, not a usable height. Propagate that failure rather than averaging or comparing it as a valid depth.</span></figcaption></figure>

Root-only testing fails here. Its child heights are three and two, only one apart; node `2` still violates the condition.

## Code

Use this node type for a local run.

```python
from dataclasses import dataclass


@dataclass
class TreeNode:
   val: int
   left: "TreeNode | None" = None
   right: "TreeNode | None" = None
```

```python
def is_balanced(root: TreeNode | None) -> bool:
   def height(node: TreeNode | None) -> int:
      if node is None:
         return 0

      left = height(node.left)
      if left < 0:
         return -1

      right = height(node.right)
      if right < 0 or abs(left - right) > 1:
         return -1
      return 1 + max(left, right)

   return height(root) >= 0
```

### Extend the Returned Contract

A nonnegative return is the exact height. It also certifies balance everywhere in that subtree.

The value `-1` means failure, not another height.

Every legitimate height is nonnegative, so this sentinel cannot collide with valid data.

Check left failure before calling right. Then check right failure and the local difference before returning a height.

Both certified children and an acceptable local difference are necessary. Ignoring a descendant's failure loses the whole-tree guarantee.

Each visited node does constant work: worst-case $O(n)$ time and $O(h)$ call-stack space for height $h$.

Recomputing heights separately at every node can take quadratic time along a chain. Deep trees can exceed Python's recursion limit.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.

**Recall:** Why propagate failure before combining heights? A failed child has no certified height in this interface.

<!-- learning:core-end -->

## Keep Height Even on Failure

**Choose this when:** The caller needs the actual height of an unbalanced tree too.

Return height and validity separately. Visit both children even when one is unbalanced; otherwise the requested height could be wrong.

```python
def height_and_balance(root):
   if root is None:
      return [0, True]
   left_height, left_ok = height_and_balance(root.left)
   right_height, right_ok = height_and_balance(root.right)
   return [
      1 + max(left_height, right_height),
      left_ok and right_ok and abs(left_height - right_height) <= 1,
   ]
```

A three-node one-sided chain returns `[3, False]`; an empty tree returns `[0, True]`.

Unlike the sentinel version, this interface preserves the aggregate after validation fails. It visits all nodes in $O(n)$ time and retains $O(h)$ stack space.

## Change the Allowed Difference

**Choose this when:** Every local height difference may be at most a supplied nonnegative tolerance.

```python
def balanced_with_tolerance(root, tolerance):
   if tolerance < 0:
      raise ValueError("nonnegative tolerance required")

   def height(node):
      if node is None:
         return 0
      left = height(node.left)
      if left < 0:
         return -1
      right = height(node.right)
      if right < 0 or abs(left - right) > tolerance:
         return -1
      return 1 + max(left, right)

   return height(root) >= 0
```

The same three-node chain passes tolerance two and fails tolerance one. Zero requires equal child heights at every node. Negative tolerance raises `ValueError`, including for an empty tree.

Only the local threshold changes. Every descendant must still pass. Worst-case time remains $O(n)$ and stack space $O(h)$.

## Transfer

A failure sentinel is useful only while it cannot be mistaken for a valid summary. If callers need the summary after failure, return validity separately instead of discarding it.
