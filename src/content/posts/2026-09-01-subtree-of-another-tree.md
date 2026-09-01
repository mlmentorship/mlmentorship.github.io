---
title: "Subtree of Another Tree"
description: "Check whether one full tree appears inside another tree."
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

> Check whether one full tree appears inside another tree.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:subtree-of-another-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="subtree-of-another-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="subtree-of-another-tree-state-title">Subtree of Another Tree: Try the full-tree equality test at each candidate node.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="subtree-of-another-tree" role="group" tabindex="0" aria-label="Subtree of Another Tree: Try the full-tree equality test at each candidate node."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether one full tree appears inside another tree.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Scan candidate roots"><div class="coding-trace-frame-heading"><span>Scan candidate roots</span><strong>The root 3 does not match subroot root 4, so search its children.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="80" y2="172" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="240" y2="172" /><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-3-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">3</text><text class="coding-trace-node-state" x="320" y="58">try</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-4-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">4</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">5</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><circle cx="80" cy="172" r="18" /><text x="80" y="176">1</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="240" cy="172" r="18" /><text x="240" y="176">2</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Match at node 4"><div class="coding-trace-frame-heading"><span>Match at node 4</span><strong>The subtree rooted at 4 has the same value and child shape.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="80" y2="172" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="240" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">3</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-4-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">4</text><text class="coding-trace-node-state" x="160" y="130">match</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">5</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-1-0"><circle cx="80" cy="172" r="18" /><text x="80" y="176">1</text><text class="coding-trace-node-state" x="80" y="202">match</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-2-0"><circle cx="240" cy="172" r="18" /><text x="240" y="176">2</text><text class="coding-trace-node-state" x="240" y="202">match</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return true"><div class="coding-trace-frame-heading"><span>Return true</span><strong>One complete matching subtree is enough.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 216" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="80" y2="172" /><line class="coding-trace-edge-line" x1="160" y1="100" x2="240" y2="172" /><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">3</text></g><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-4-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">4</text><text class="coding-trace-node-state" x="160" y="130">subtree</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-5-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">5</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><circle cx="80" cy="172" r="18" /><text x="80" y="176">1</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-2-0"><circle cx="240" cy="172" r="18" /><text x="240" y="176">2</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Scan candidate roots</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Match at node 4</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return true</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Try the full-tree equality test at each candidate node.</p></div><figcaption><strong>Read it this way:</strong> The root 3 does not match subroot root 4, so search its children. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** DFS plus Same Tree.

**Simple idea:** At each node, check whether the trees are the same from that point. If not,
search the left and right subtrees.

```python
def is_subtree(root: TreeNode | None, subroot: TreeNode | None) -> bool:
   def same(first: TreeNode | None, second: TreeNode | None) -> bool:
      if first is None or second is None:
         return first is second
      return (
         first.val == second.val
         and same(first.left, second.left)
         and same(first.right, second.right)
      )

   if subroot is None:
      return True
   if root is None:
      return False
   return (
      same(root, subroot)
      or is_subtree(root.left, subroot)
      or is_subtree(root.right, subroot)
   )
```

**Cost:** $O(mn)$ time in the worst case and $O(h)$ call-stack space.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
