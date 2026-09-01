---
title: "Construct Tree From Preorder and Inorder Traversal"
description: "Rebuild a binary tree from its preorder and inorder value lists. Values are unique."
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

> Rebuild a binary tree from its preorder and inorder value lists. Values are unique.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:construct-tree-from-preorder-and-inorder-traversal-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="construct-tree-from-preorder-and-inorder-traversal-state-title"><p class="visual-kicker">A child answer returns upward</p><p class="visual-title" id="construct-tree-from-preorder-and-inorder-traversal-state-title">Construct Tree From Preorder and Inorder Traversal: Solve a node by asking each child for one complete fact</p><div class="coding-visual coding-visual--tree" data-coding-visual data-coding-mode="tree" data-coding-slug="construct-tree-from-preorder-and-inorder-traversal" role="group" aria-label="Construct Tree From Preorder and Inorder Traversal: preorder gives root 3; inorder splits [9] from [15,20,7]. A returned value completely summarizes the subtree below its node."><div class="coding-visual-example"><span>Concrete trace</span><strong>preorder gives root 3; inorder splits [9] from [15,20,7]</strong></div><div class="coding-visual-sketch coding-visual-sketch--tree"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">parent</span><div class="coding-sketch-branch"><span class="coding-sketch-node">left fact</span><span class="coding-sketch-node">right fact</span></div></div><p class="coding-sketch-note">children return compact facts; the parent combines them</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Enter</span><strong>current node</strong><small>The call owns one subtree.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Ask</span><strong>left / right</strong><small>Each child returns its subtree fact.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Combine</span><strong>node rule</strong><small>Use the child facts to score or validate here.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Return</span><strong>one useful value</strong><small>Pass only what the parent can still use.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>A returned value completely summarizes the subtree below its node.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the tree bottom-up even when the code is recursive. A parent does not need every descendant, only the compact fact each child promises to return. For this problem, hold onto the concrete trace: preorder gives root 3; inorder splits [9] from [15,20,7].</figcaption></figure>

**Pattern:** Preorder chooses roots and inorder splits child ranges.

**Simple idea:** The next preorder value is the current root. Its inorder position separates
the left and right subtrees. A map makes that position lookup constant time.

```python
def build_tree(preorder: list[int], inorder: list[int]) -> TreeNode | None:
   positions = {value: index for index, value in enumerate(inorder)}
   preorder_index = 0

   def build(left: int, right: int) -> TreeNode | None:
      nonlocal preorder_index
      if left > right:
         return None

      value = preorder[preorder_index]
      preorder_index += 1
      node = TreeNode(value)
      middle = positions[value]
      node.left = build(left, middle - 1)
      node.right = build(middle + 1, right)
      return node

   return build(0, len(inorder) - 1)
```

**Cost:** $O(n)$ time and $O(n)$ space.
