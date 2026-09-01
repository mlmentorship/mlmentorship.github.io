---
title: "Validate Binary Search Tree"
description: "Check whether every node follows all BST ordering rules."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Mixed"
priority: "Core"
aliases: []
prerequisites: []
---

> Check whether every node follows all BST ordering rules.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:validate-binary-search-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="validate-binary-search-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="validate-binary-search-tree-state-title">Validate Binary Search Tree: Pass the full inherited lower and upper bounds down each tree branch.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="validate-binary-search-tree" role="group" aria-label="Validate Binary Search Tree: Pass the full inherited lower and upper bounds down each tree branch."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether every node follows all BST ordering rules.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Set the root bounds"><div class="coding-trace-frame-heading"><span>Set the root bounds</span><strong>Root 5 must lie between negative and positive infinity.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-focus"><span>5</span><small>bounds (-inf,inf)</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>1</span></span><span class="coding-trace-tree-node"><span>7</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>4</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Carry an ancestor bound"><div class="coding-trace-frame-heading"><span>Carry an ancestor bound</span><strong>Node 4 is in the right subtree of 5, so its lower bound is 5.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>5</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>1</span></span><span class="coding-trace-tree-node"><span>7</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node trace-tone-warning"><span>4</span><small>4 not &gt; 5</small></span><span class="coding-trace-tree-node"><span>-</span></span></div></div><div class="coding-trace-meta"><span><b>bounds</b>4 must be &gt; 5</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Reject the tree"><div class="coding-trace-frame-heading"><span>Reject the tree</span><strong>A parent-only check would miss this violation; inherited bounds catch it.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>5</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>1</span></span><span class="coding-trace-tree-node"><span>7</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node trace-tone-output"><span>4</span><small>invalid</small></span><span class="coding-trace-tree-node"><span>-</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>false</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Set the root bounds</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Carry an ancestor bound</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Reject the tree</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Pass the full inherited lower and upper bounds down each tree branch.</p></div><figcaption><strong>Read it this way:</strong> Root 5 must lie between negative and positive infinity. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** DFS with valid lower and upper bounds.

**Simple idea:** A node in a left subtree must be below every ancestor bound, not only its
parent. Pass the allowed value range down the tree.

```python
def is_valid_bst(root: TreeNode | None) -> bool:
   def valid(node: TreeNode | None, low: float, high: float) -> bool:
      if node is None:
         return True
      if not low < node.val < high:
         return False
      return valid(node.left, low, node.val) and valid(node.right, node.val, high)

   return valid(root, float("-inf"), float("inf"))
```

**Cost:** $O(n)$ time and $O(h)$ call-stack space.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
