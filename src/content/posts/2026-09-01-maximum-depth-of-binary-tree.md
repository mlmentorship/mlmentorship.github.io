---
title: "Maximum Depth of Binary Tree"
description: "Find the number of nodes on the longest root-to-leaf path."
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

> Find the number of nodes on the longest root-to-leaf path.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:maximum-depth-of-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="maximum-depth-of-binary-tree-state-title"><p class="visual-kicker">A child answer returns upward</p><p class="visual-title" id="maximum-depth-of-binary-tree-state-title">Maximum Depth of Binary Tree: Solve a node by asking each child for one complete fact</p><div class="coding-visual coding-visual--tree" data-coding-visual data-coding-mode="tree" data-coding-slug="maximum-depth-of-binary-tree" role="group" aria-label="Maximum Depth of Binary Tree: leaf returns 1; parent returns 1 + max(left_depth, right_depth). A returned value completely summarizes the subtree below its node."><div class="coding-visual-example"><span>Concrete trace</span><strong>leaf returns 1; parent returns 1 + max(left_depth, right_depth)</strong></div><div class="coding-visual-sketch coding-visual-sketch--tree"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">parent</span><div class="coding-sketch-branch"><span class="coding-sketch-node">left fact</span><span class="coding-sketch-node">right fact</span></div></div><p class="coding-sketch-note">children return compact facts; the parent combines them</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Enter</span><strong>current node</strong><small>The call owns one subtree.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Ask</span><strong>left / right</strong><small>Each child returns its subtree fact.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Combine</span><strong>node rule</strong><small>Use the child facts to score or validate here.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Return</span><strong>one useful value</strong><small>Pass only what the parent can still use.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>A returned value completely summarizes the subtree below its node.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the tree bottom-up even when the code is recursive. A parent does not need every descendant, only the compact fact each child promises to return. For this problem, hold onto the concrete trace: leaf returns 1; parent returns 1 + max(left_depth, right_depth).</figcaption></figure>

**Pattern:** Bottom-up tree DFS.

**Simple idea:** A node's depth is one plus the larger depth from its two children.

```python
def max_depth(root: TreeNode | None) -> int:
   if root is None:
      return 0
   return 1 + max(max_depth(root.left), max_depth(root.right))
```

**Cost:** $O(n)$ time and $O(h)$ call-stack space, where $h$ is tree height.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
