---
title: "Kth Smallest Element in a BST"
description: "Return the `k`th smallest tree value."
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

> Return the `k`th smallest tree value.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:kth-smallest-element-in-a-bst-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="kth-smallest-element-in-a-bst-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="kth-smallest-element-in-a-bst-state-title">Kth Smallest Element in a BST: Inorder traversal visits BST nodes in ascending order, so stop at the kth visit.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="kth-smallest-element-in-a-bst" role="group" aria-label="Kth Smallest Element in a BST: Inorder traversal visits BST nodes in ascending order, so stop at the kth visit."><div class="coding-visual-example"><span>Input and goal</span><strong>Return the `k`th smallest tree value.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Push the left spine"><div class="coding-trace-frame-heading"><span>Push the left spine</span><strong>Start by pushing 3, then 1.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-state"><span>3</span><small>stack</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node trace-tone-focus"><span>1</span><small>stack</small></span><span class="coding-trace-tree-node"><span>4</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>2</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Visit in order"><div class="coding-trace-frame-heading"><span>Visit in order</span><strong>Pop 1 first, then 2, then 3. The first visit is the smallest.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>3</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node trace-tone-focus"><span>1</span><small>visit 1</small></span><span class="coding-trace-tree-node"><span>4</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node trace-tone-state"><span>2</span><small>visit 2</small></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Stop at k"><div class="coding-trace-frame-heading"><span>Stop at k</span><strong>For k=1, return node 1 immediately.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>3</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node trace-tone-output"><span>1</span><small>kth</small></span><span class="coding-trace-tree-node"><span>4</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>2</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>1</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Push the left spine</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Visit in order</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Stop at k</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Inorder traversal visits BST nodes in ascending order, so stop at the kth visit.</p></div><figcaption><strong>Read it this way:</strong> Start by pushing 3, then 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Iterative inorder traversal.

**Simple idea:** Inorder visits BST values from smallest to largest. Stop at the `k`th visited
node.

```python
def kth_smallest(root: TreeNode | None, k: int) -> int:
   stack: list[TreeNode] = []

   while root or stack:
      while root:
         stack.append(root)
         root = root.left
      root = stack.pop()
      k -= 1
      if k == 0:
         return root.val
      root = root.right
   raise ValueError("k is larger than the tree")
```

**Cost:** $O(h + k)$ time and $O(h)$ space.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
