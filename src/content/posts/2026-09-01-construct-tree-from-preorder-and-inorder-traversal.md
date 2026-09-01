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
<figure class="learning-figure coding-visual-figure" aria-labelledby="construct-tree-from-preorder-and-inorder-traversal-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="construct-tree-from-preorder-and-inorder-traversal-state-title">Construct Tree From Preorder and Inorder Traversal: Preorder gives the next root; inorder splits the left and right ranges.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="construct-tree-from-preorder-and-inorder-traversal" role="group" aria-label="Construct Tree From Preorder and Inorder Traversal: Preorder gives the next root; inorder splits the left and right ranges."><div class="coding-visual-example"><span>Input and goal</span><strong>Rebuild a binary tree from its preorder and inorder value lists. Values are unique.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Choose the root"><div class="coding-trace-frame-heading"><span>Choose the root</span><strong>Preorder starts with 3. Inorder places 3 between 9 and 15,20,7.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">preorder next</th><th scope="col">inorder left</th><th scope="col">root</th><th scope="col">inorder right</th></tr></thead><tbody><tr><td class="">3</td><td class="">[9]</td><td class="is-active">3</td><td class="">[15,20,7]</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Recurse on ranges"><div class="coding-trace-frame-heading"><span>Recurse on ranges</span><strong>The next preorder values become roots of the left and right ranges.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>3</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node trace-tone-state"><span>9</span><small>left range</small></span><span class="coding-trace-tree-node trace-tone-focus"><span>20</span><small>right range</small></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>15</span></span><span class="coding-trace-tree-node"><span>7</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return the tree"><div class="coding-trace-frame-heading"><span>Return the tree</span><strong>Every inorder range is reconstructed with one preorder root.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-output"><span>3</span><small>root</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>9</span></span><span class="coding-trace-tree-node"><span>20</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>15</span></span><span class="coding-trace-tree-node"><span>7</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>tree rebuilt</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Choose the root</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Recurse on ranges</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the tree</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Preorder gives the next root; inorder splits the left and right ranges.</p></div><figcaption><strong>Read it this way:</strong> Preorder starts with 3. Inorder places 3 between 9 and 15,20,7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
