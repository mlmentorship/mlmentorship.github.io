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
<figure class="learning-figure coding-visual-figure" aria-labelledby="lowest-common-ancestor-in-a-bst-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="lowest-common-ancestor-in-a-bst-state-title">Lowest Common Ancestor in a BST: BST ordering tells whether both targets lie left, right, or split at the current node.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="lowest-common-ancestor-in-a-bst" role="group" tabindex="0" aria-label="Lowest Common Ancestor in a BST: BST ordering tells whether both targets lie left, right, or split at the current node."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the lowest node whose subtree contains both target nodes.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Start at 6"><div class="coding-trace-frame-heading"><span>Start at 6</span><strong>Targets 2 and 8 lie on opposite sides of 6.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 144" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-6-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">6</text><text class="coding-trace-node-state" x="320" y="58">split</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">2</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-8-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">8</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Stop at the split"><div class="coding-trace-frame-heading"><span>Stop at the split</span><strong>If both targets were left or right, descend; here 6 is the first split.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 144" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-6-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">6</text><text class="coding-trace-node-state" x="320" y="58">ancestor</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">2</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-8-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">8</text></g></svg></div><div class="coding-trace-meta"><span><b>path</b>2 &lt; 6 &lt; 8</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the ancestor"><div class="coding-trace-frame-heading"><span>Return the ancestor</span><strong>Node 6 is the lowest node whose subtree contains both targets.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 144" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-6-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">6</text><text class="coding-trace-node-state" x="320" y="58">LCA</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">2</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-8-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">8</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>6</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start at 6</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Stop at the split</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the ancestor</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>BST ordering tells whether both targets lie left, right, or split at the current node.</p></div><figcaption><strong>Read it this way:</strong> Targets 2 and 8 lie on opposite sides of 6. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
