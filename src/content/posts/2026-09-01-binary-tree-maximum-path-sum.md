---
title: "Binary Tree Maximum Path Sum"
description: "Find the largest sum of any connected path in a binary tree."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

> Find the largest sum of any connected path in a binary tree.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:binary-tree-maximum-path-sum-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-tree-maximum-path-sum-state-title"><p class="visual-kicker">A child answer returns upward</p><p class="visual-title" id="binary-tree-maximum-path-sum-state-title">Binary Tree Maximum Path Sum: Solve a node by asking each child for one complete fact</p><div class="coding-visual coding-visual--tree" data-coding-visual data-coding-mode="tree" data-coding-slug="binary-tree-maximum-path-sum" role="group" aria-label="Binary Tree Maximum Path Sum: a node may return one child upward but score both children locally. A returned value completely summarizes the subtree below its node."><div class="coding-visual-example"><span>Concrete trace</span><strong>a node may return one child upward but score both children locally</strong></div><div class="coding-visual-sketch coding-visual-sketch--tree"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">parent</span><div class="coding-sketch-branch"><span class="coding-sketch-node">left fact</span><span class="coding-sketch-node">right fact</span></div></div><p class="coding-sketch-note">children return compact facts; the parent combines them</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Enter</span><strong>current node</strong><small>The call owns one subtree.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Ask</span><strong>left / right</strong><small>Each child returns its subtree fact.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Combine</span><strong>node rule</strong><small>Use the child facts to score or validate here.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Return</span><strong>one useful value</strong><small>Pass only what the parent can still use.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>A returned value completely summarizes the subtree below its node.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the tree bottom-up even when the code is recursive. A parent does not need every descendant, only the compact fact each child promises to return. For this problem, hold onto the concrete trace: a node may return one child upward but score both children locally.</figcaption></figure>

**Pattern:** Return one branch, score two branches.

**Simple idea:** A parent path can use only one branch from a child. A path whose highest
node is the current node can use both branches. Return one branch, but use both when
updating
the full answer.

```python
def max_path_sum(root: TreeNode | None) -> int:
   if root is None:
      return 0

   best = root.val

   def one_branch(node: TreeNode | None) -> int:
      nonlocal best
      if node is None:
         return 0

      left = max(0, one_branch(node.left))
      right = max(0, one_branch(node.right))
      best = max(best, node.val + left + right)
      return node.val + max(left, right)

   one_branch(root)
   return best
```

**Cost:** $O(n)$ time and $O(h)$ call-stack space.
