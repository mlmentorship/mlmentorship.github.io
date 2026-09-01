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
<figure class="learning-figure coding-visual-figure" aria-labelledby="kth-smallest-element-in-a-bst-state-title"><p class="visual-kicker">A child answer returns upward</p><p class="visual-title" id="kth-smallest-element-in-a-bst-state-title">Kth Smallest Element in a BST: Solve a node by asking each child for one complete fact</p><div class="coding-visual coding-visual--tree" data-coding-visual data-coding-mode="tree" data-coding-slug="kth-smallest-element-in-a-bst" role="group" aria-label="Kth Smallest Element in a BST: inorder visits 1,2,3,...; stop exactly at the kth visit. A returned value completely summarizes the subtree below its node."><div class="coding-visual-example"><span>Concrete trace</span><strong>inorder visits 1,2,3,...; stop exactly at the kth visit</strong></div><div class="coding-visual-sketch coding-visual-sketch--tree"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">parent</span><div class="coding-sketch-branch"><span class="coding-sketch-node">left fact</span><span class="coding-sketch-node">right fact</span></div></div><p class="coding-sketch-note">children return compact facts; the parent combines them</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Enter</span><strong>current node</strong><small>The call owns one subtree.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Ask</span><strong>left / right</strong><small>Each child returns its subtree fact.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Combine</span><strong>node rule</strong><small>Use the child facts to score or validate here.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Return</span><strong>one useful value</strong><small>Pass only what the parent can still use.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>A returned value completely summarizes the subtree below its node.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the tree bottom-up even when the code is recursive. A parent does not need every descendant, only the compact fact each child promises to return. For this problem, hold onto the concrete trace: inorder visits 1,2,3,...; stop exactly at the kth visit.</figcaption></figure>

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
