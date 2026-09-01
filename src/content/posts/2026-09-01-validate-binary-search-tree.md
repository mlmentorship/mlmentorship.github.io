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
<figure class="learning-figure coding-visual-figure" aria-labelledby="validate-binary-search-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="validate-binary-search-tree-state-title">Validate Binary Search Tree: Pass the full inherited lower and upper bounds down each tree branch.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="validate-binary-search-tree" role="group" tabindex="0" aria-label="Validate Binary Search Tree: Pass the full inherited lower and upper bounds down each tree branch."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether every node follows all BST ordering rules.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Set the root bounds"><div class="coding-trace-frame-heading"><span>Set the root bounds</span><strong>Root 5 must lie between negative and positive infinity.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="400" y2="172" /><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-5-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">5</text><text class="coding-trace-node-state" x="320" y="58">bounds (-inf,inf)</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">1</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">7</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><circle cx="400" cy="172" r="18" /><text x="400" y="176">4</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Carry an ancestor bound"><div class="coding-trace-frame-heading"><span>Carry an ancestor bound</span><strong>Node 4 is in the right subtree of 5, so its lower bound is 5.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="400" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">5</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">1</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">7</text></g><g class="coding-trace-tree-node trace-tone-warning" data-motion-key="tree-node-4-0"><circle cx="400" cy="172" r="18" /><text x="400" y="176">4</text><text class="coding-trace-node-state" x="400" y="202">4 not &gt; 5</text></g></svg></div><div class="coding-trace-meta"><span><b>bounds</b>4 must be &gt; 5</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Reject the tree"><div class="coding-trace-frame-heading"><span>Reject the tree</span><strong>A parent-only check would miss this violation; inherited bounds catch it.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="400" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">5</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">1</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">7</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-4-0"><circle cx="400" cy="172" r="18" /><text x="400" y="176">4</text><text class="coding-trace-node-state" x="400" y="202">invalid</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>false</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Set the root bounds</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Carry an ancestor bound</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Reject the tree</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Pass the full inherited lower and upper bounds down each tree branch.</p></div><figcaption><strong>Read it this way:</strong> Root 5 must lie between negative and positive infinity. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
