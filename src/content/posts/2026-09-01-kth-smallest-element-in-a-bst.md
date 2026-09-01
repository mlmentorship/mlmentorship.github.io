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
<figure class="learning-figure coding-visual-figure" aria-labelledby="kth-smallest-element-in-a-bst-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="kth-smallest-element-in-a-bst-state-title">Kth Smallest Element in a BST: Inorder traversal visits BST nodes in ascending order, so stop at the kth visit.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="kth-smallest-element-in-a-bst" role="group" tabindex="0" aria-label="Kth Smallest Element in a BST: Inorder traversal visits BST nodes in ascending order, so stop at the kth visit."><div class="coding-visual-example"><span>Input and goal</span><strong>Return the `k`th smallest tree value.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Push the left spine"><div class="coding-trace-frame-heading"><span>Push the left spine</span><strong>Start by pushing 3, then 1.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="240" y2="172" /><g class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-3-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">3</text><text class="coding-trace-node-state" x="320" y="58">stack</text></g><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-1-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">1</text><text class="coding-trace-node-state" x="160" y="130">stack</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">4</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="240" cy="172" r="18" /><text x="240" y="176">2</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Visit in order"><div class="coding-trace-frame-heading"><span>Visit in order</span><strong>Pop 1 first, then 2, then 3. The first visit is the smallest.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="240" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">3</text></g><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-1-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">1</text><text class="coding-trace-node-state" x="160" y="130">visit 1</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">4</text></g><g class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-2-0"><circle cx="240" cy="172" r="18" /><text x="240" y="176">2</text><text class="coding-trace-node-state" x="240" y="202">visit 2</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Stop at k"><div class="coding-trace-frame-heading"><span>Stop at k</span><strong>For k=1, return node 1 immediately.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="240" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">3</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-1-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">1</text><text class="coding-trace-node-state" x="160" y="130">kth</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">4</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="240" cy="172" r="18" /><text x="240" y="176">2</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>1</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Push the left spine</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Visit in order</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Stop at k</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Inorder traversal visits BST nodes in ascending order, so stop at the kth visit.</p></div><figcaption><strong>Read it this way:</strong> Start by pushing 3, then 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
