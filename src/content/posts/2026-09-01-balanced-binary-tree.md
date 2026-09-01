---
title: "Balanced Binary Tree"
description: "Check whether the child heights at every node differ by at most one."
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

> Check whether the child heights at every node differ by at most one.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:balanced-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="balanced-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="balanced-binary-tree-state-title">Balanced Binary Tree: Return a failure sentinel as soon as child heights differ by more than one.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="balanced-binary-tree" role="group" tabindex="0" aria-label="Balanced Binary Tree: Return a failure sentinel as soon as child heights differ by more than one."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether the child heights at every node differ by at most one.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Compute child heights"><div class="coding-trace-frame-heading"><span>Compute child heights</span><strong>A chain gives the left subtree height 2 and the right height 0.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="160" y2="172" /><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-1-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">1</text><text class="coding-trace-node-state" x="320" y="58">check</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">2</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="160" cy="172" r="18" /><text x="160" y="176">3</text></g></svg></div><div class="coding-trace-meta"><span><b>detail</b>left=2, right=0</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Propagate failure"><div class="coding-trace-frame-heading"><span>Propagate failure</span><strong>The difference is 2, so this subtree returns -1.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="160" y2="172" /><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-1-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">1</text><text class="coding-trace-node-state" x="320" y="58">unbalanced</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">2</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="160" cy="172" r="18" /><text x="160" y="176">3</text></g></svg></div><div class="coding-trace-meta"><span><b>detail</b>return -1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Answer false"><div class="coding-trace-frame-heading"><span>Answer false</span><strong>The root sees the sentinel and stops.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="160" y2="172" /><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-1-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">1</text><text class="coding-trace-node-state" x="320" y="58">false</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">2</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="160" cy="172" r="18" /><text x="160" y="176">3</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>false</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Compute child heights</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Propagate failure</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Answer false</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Return a failure sentinel as soon as child heights differ by more than one.</p></div><figcaption><strong>Read it this way:</strong> A chain gives the left subtree height 2 and the right height 0. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Bottom-up tree DFS with an error value.

**Simple idea:** Return the subtree height when it is balanced. Return `-1` when it is not.
Once a child returns `-1`, pass it upward without more work.

```python
def is_balanced(root: TreeNode | None) -> bool:
   def height(node: TreeNode | None) -> int:
      if node is None:
         return 0

      left = height(node.left)
      if left < 0:
         return -1

      right = height(node.right)
      if right < 0 or abs(left - right) > 1:
         return -1
      return 1 + max(left, right)

   return height(root) >= 0
```

**Cost:** $O(n)$ time and $O(h)$ call-stack space.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
