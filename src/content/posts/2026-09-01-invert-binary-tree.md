---
title: "Invert Binary Tree"
description: "Swap the left and right children at every node."
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

> Swap the left and right children at every node.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:invert-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="invert-binary-tree-state-title"><p class="visual-kicker">A child answer returns upward</p><p class="visual-title" id="invert-binary-tree-state-title">Invert Binary Tree: Solve a node by asking each child for one complete fact</p><div class="coding-visual coding-visual--tree" data-coding-visual data-coding-mode="tree" data-coding-slug="invert-binary-tree" role="group" aria-label="Invert Binary Tree: at every node, swap left and right before returning upward. A returned value completely summarizes the subtree below its node."><div class="coding-visual-example"><span>Concrete trace</span><strong>at every node, swap left and right before returning upward</strong></div><div class="coding-visual-sketch coding-visual-sketch--tree"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">parent</span><div class="coding-sketch-branch"><span class="coding-sketch-node">left fact</span><span class="coding-sketch-node">right fact</span></div></div><p class="coding-sketch-note">children return compact facts; the parent combines them</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Enter</span><strong>current node</strong><small>The call owns one subtree.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Ask</span><strong>left / right</strong><small>Each child returns its subtree fact.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Combine</span><strong>node rule</strong><small>Use the child facts to score or validate here.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Return</span><strong>one useful value</strong><small>Pass only what the parent can still use.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>A returned value completely summarizes the subtree below its node.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the tree bottom-up even when the code is recursive. A parent does not need every descendant, only the compact fact each child promises to return. For this problem, hold onto the concrete trace: at every node, swap left and right before returning upward.</figcaption></figure>

**Pattern:** Tree DFS.

**Simple idea:** Invert both child trees, then place the old right result on the left and the
old left result on the right.

```python
def invert_tree(root: TreeNode | None) -> TreeNode | None:
   if root is None:
      return None
   root.left, root.right = invert_tree(root.right), invert_tree(root.left)
   return root
```

**Cost:** $O(n)$ time and $O(h)$ call-stack space.
