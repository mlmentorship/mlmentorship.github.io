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
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-tree-maximum-path-sum-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="binary-tree-maximum-path-sum-state-title">Binary Tree Maximum Path Sum: A node returns one child branch upward but can score both child branches locally.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="binary-tree-maximum-path-sum" role="group" tabindex="0" aria-label="Binary Tree Maximum Path Sum: A node returns one child branch upward but can score both child branches locally."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the largest sum of any connected path in a binary tree.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Return one branch"><div class="coding-trace-frame-heading"><span>Return one branch</span><strong>At node 20, the larger child contribution is 15, while the full path can use 15 and 7.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="400" y2="172" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="560" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node--10-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">-10</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">9</text></g><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">20</text><text class="coding-trace-node-state" x="480" y="130">score 42</text></g><g class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node-15-0"><circle cx="400" cy="172" r="18" /><text x="400" y="176">15</text><text class="coding-trace-node-state" x="400" y="202">return 15</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><circle cx="560" cy="172" r="18" /><text x="560" y="176">7</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Reject negative branches"><div class="coding-trace-frame-heading"><span>Reject negative branches</span><strong>A negative child contribution is replaced by zero before combining.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="400" y2="172" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="560" y2="172" /><g class="coding-trace-tree-node trace-tone-state" data-motion-key="tree-node--10-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">-10</text><text class="coding-trace-node-state" x="320" y="58">left 0</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">9</text></g><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-20-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">20</text><text class="coding-trace-node-state" x="480" y="130">both children</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-15-0"><circle cx="400" cy="172" r="18" /><text x="400" y="176">15</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-7-0"><circle cx="560" cy="172" r="18" /><text x="560" y="176">7</text></g></svg></div><div class="coding-trace-meta"><span><b>formula</b>20 + 15 + 7 = 42</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Update global best"><div class="coding-trace-frame-heading"><span>Update global best</span><strong>The path 15 -&gt; 20 -&gt; 7 has the maximum sum 42.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="400" y2="172" /><line class="coding-trace-edge-line" x1="480" y1="100" x2="560" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node--10-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">-10</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-9-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">9</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-20-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">20</text><text class="coding-trace-node-state" x="480" y="130">best path</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-15-0"><circle cx="400" cy="172" r="18" /><text x="400" y="176">15</text><text class="coding-trace-node-state" x="400" y="202">best path</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-7-0"><circle cx="560" cy="172" r="18" /><text x="560" y="176">7</text><text class="coding-trace-node-state" x="560" y="202">best path</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>42</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Return one branch</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Reject negative branches</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Update global best</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A node returns one child branch upward but can score both child branches locally.</p></div><figcaption><strong>Read it this way:</strong> At node 20, the larger child contribution is 15, while the full path can use 15 and 7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
