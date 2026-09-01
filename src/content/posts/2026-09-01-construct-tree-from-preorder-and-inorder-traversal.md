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
<figure class="learning-figure coding-visual-figure" aria-labelledby="construct-tree-from-preorder-and-inorder-traversal-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="construct-tree-from-preorder-and-inorder-traversal-state-title">Construct Tree From Preorder and Inorder Traversal: Preorder gives the next root; inorder splits the left and right ranges.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="construct-tree-from-preorder-and-inorder-traversal" role="group" tabindex="0" aria-label="Construct Tree From Preorder and Inorder Traversal: Preorder gives the next root; inorder splits the left and right ranges."><div class="coding-visual-example"><span>Input and goal</span><strong>Rebuild a binary tree from its preorder and inorder value lists. Values are unique.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Choose the root"><div class="coding-trace-frame-heading"><span>Choose the root</span><strong>Preorder starts with 3. Inorder places 3 between 9 and 15,20,7.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">preorder next</th><th scope="col">inorder left</th><th scope="col">root</th><th scope="col">inorder right</th></tr></thead><tbody><tr><td class="">3</td><td class="">[9]</td><td class="is-active">3</td><td class="">[15,20,7]</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Recurse on ranges"><div class="coding-trace-frame-heading"><span>Recurse on ranges</span><strong>The next preorder values become roots of the left and right ranges.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="400" y2="172" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="560" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">3</text></g><g class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-9-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">9</text><text class="coding-trace-node-state" x="160" y="130">left range</text></g><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">20</text><text class="coding-trace-node-state" x="480" y="130">right range</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><circle cx="400" cy="172" r="18" /><text x="400" y="176">15</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><circle cx="560" cy="172" r="18" /><text x="560" y="176">7</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the tree"><div class="coding-trace-frame-heading"><span>Return the tree</span><strong>Every inorder range is reconstructed with one preorder root.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="400" y2="172" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="560" y2="172" /><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-3-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">3</text><text class="coding-trace-node-state" x="320" y="58">root</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">9</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-20-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">20</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><circle cx="400" cy="172" r="18" /><text x="400" y="176">15</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><circle cx="560" cy="172" r="18" /><text x="560" y="176">7</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>tree rebuilt</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Choose the root</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Recurse on ranges</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the tree</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Preorder gives the next root; inorder splits the left and right ranges.</p></div><figcaption><strong>Read it this way:</strong> Preorder starts with 3. Inorder places 3 between 9 and 15,20,7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
