---
title: "Balanced Binary Tree"
description: "Check whether the child heights at every node differ by at most one."
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

> Check whether the child heights at every node differ by at most one.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:balanced-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="balanced-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="balanced-binary-tree-state-title">Balanced Binary Tree: Return a failure sentinel as soon as child heights differ by more than one.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="balanced-binary-tree" role="group" aria-label="Balanced Binary Tree: Return a failure sentinel as soon as child heights differ by more than one."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether the child heights at every node differ by at most one.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Compute child heights"><div class="coding-trace-frame-heading"><span>Compute child heights</span><strong>A chain gives the left subtree height 2 and the right height 0.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-focus"><span>1</span><small>check</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>2</span></span><span class="coding-trace-tree-node"><span>-</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>3</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div><div class="coding-trace-meta"><span><b>detail</b>left=2, right=0</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Propagate failure"><div class="coding-trace-frame-heading"><span>Propagate failure</span><strong>The difference is 2, so this subtree returns -1.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-focus"><span>1</span><small>unbalanced</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>2</span></span><span class="coding-trace-tree-node"><span>-</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>3</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div><div class="coding-trace-meta"><span><b>detail</b>return -1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Answer false"><div class="coding-trace-frame-heading"><span>Answer false</span><strong>The root sees the sentinel and stops.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-output"><span>1</span><small>false</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>2</span></span><span class="coding-trace-tree-node"><span>-</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>3</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>false</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Compute child heights</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Propagate failure</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Answer false</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Return a failure sentinel as soon as child heights differ by more than one.</p></div><figcaption><strong>Read it this way:</strong> A chain gives the left subtree height 2 and the right height 0. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Bottom-up tree DFS with an error value.

**Simple idea:** Return the subtree height when it is balanced. Return `-1` when it is not.
Once a child returns `-1`, pass it upward without more work.

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

**Cost:** $O(n)$ time and $O(h)$ call-stack space.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
