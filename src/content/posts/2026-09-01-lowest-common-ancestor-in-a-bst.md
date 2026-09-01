---
title: "Lowest Common Ancestor in a BST"
description: "Find the lowest node whose subtree contains both target nodes."
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

> Find the lowest node whose subtree contains both target nodes.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:lowest-common-ancestor-in-a-bst-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="lowest-common-ancestor-in-a-bst-state-title"><p class="visual-kicker">A child answer returns upward</p><p class="visual-title" id="lowest-common-ancestor-in-a-bst-state-title">Lowest Common Ancestor in a BST: Solve a node by asking each child for one complete fact</p><div class="coding-visual coding-visual--tree" data-coding-visual data-coding-mode="tree" data-coding-slug="lowest-common-ancestor-in-a-bst" role="group" aria-label="Lowest Common Ancestor in a BST: if both targets are left, go left; if both right, go right; otherwise stop. A returned value completely summarizes the subtree below its node."><div class="coding-visual-example"><span>Concrete trace</span><strong>if both targets are left, go left; if both right, go right; otherwise stop</strong></div><div class="coding-visual-sketch coding-visual-sketch--tree"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">parent</span><div class="coding-sketch-branch"><span class="coding-sketch-node">left fact</span><span class="coding-sketch-node">right fact</span></div></div><p class="coding-sketch-note">children return compact facts; the parent combines them</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Enter</span><strong>current node</strong><small>The call owns one subtree.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Ask</span><strong>left / right</strong><small>Each child returns its subtree fact.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Combine</span><strong>node rule</strong><small>Use the child facts to score or validate here.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Return</span><strong>one useful value</strong><small>Pass only what the parent can still use.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>A returned value completely summarizes the subtree below its node.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the tree bottom-up even when the code is recursive. A parent does not need every descendant, only the compact fact each child promises to return. For this problem, hold onto the concrete trace: if both targets are left, go left; if both right, go right; otherwise stop.</figcaption></figure>

**Pattern:** Use BST ordering to choose one branch.

**Simple idea:** If both targets are smaller, go left. If both are larger, go right. When
they split across the current value, the current node is their lowest common ancestor.

```python
def lowest_common_ancestor_bst(
   root: TreeNode, first: TreeNode, second: TreeNode
) -> TreeNode:
   while True:
      if first.val < root.val and second.val < root.val:
         root = root.left
      elif first.val > root.val and second.val > root.val:
         root = root.right
      else:
         return root
```

**Cost:** $O(h)$ time and $O(1)$ space.
