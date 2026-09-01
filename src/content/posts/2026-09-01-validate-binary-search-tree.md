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
<figure class="learning-figure coding-visual-figure" aria-labelledby="validate-binary-search-tree-state-title"><p class="visual-kicker">A shrinking answer space</p><p class="visual-title" id="validate-binary-search-tree-state-title">Validate Binary Search Tree: Discard a half only after a yes-or-no test</p><div class="coding-visual coding-visual--binary" data-coding-visual data-coding-mode="binary" data-coding-slug="validate-binary-search-tree" role="group" aria-label="Validate Binary Search Tree: a node 4 in the right subtree of 5 violates the inherited lower bound 5. The answer never leaves the current low-to-high interval."><div class="coding-visual-example"><span>Concrete trace</span><strong>a node 4 in the right subtree of 5 violates the inherited lower bound 5</strong></div><div class="coding-visual-sketch coding-visual-sketch--binary"><div class="coding-sketch-array"><span class="coding-sketch-pointer">lo</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell coding-sketch-cell--active">mid</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-pointer">hi</span></div><p class="coding-sketch-note">probe the middle, then discard the side the predicate rules out</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Bound</span><strong>lo ... hi</strong><small>Every possible answer is inside this interval.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Probe</span><strong>mid</strong><small>Test the middle value or candidate answer.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Decide</span><strong>predicate</strong><small>The monotone result says which side can survive.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Keep</span><strong>one half</strong><small>Move one boundary and preserve the answer.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The answer never leaves the current low-to-high interval.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the interval as a promise: everything outside it is already impossible. The midpoint is useful only because the predicate is monotone. For this problem, hold onto the concrete trace: a node 4 in the right subtree of 5 violates the inherited lower bound 5.</figcaption></figure>

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
