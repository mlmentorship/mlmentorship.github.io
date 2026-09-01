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
<figure class="learning-figure coding-visual-figure" aria-labelledby="subtree-of-another-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="subtree-of-another-tree-state-title">Subtree of Another Tree: Try the full-tree equality test at each candidate node.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="subtree-of-another-tree" role="group" aria-label="Subtree of Another Tree: Try the full-tree equality test at each candidate node."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether one full tree appears inside another tree.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Scan candidate roots"><div class="coding-trace-frame-heading"><span>Scan candidate roots</span><strong>The root 3 does not match subroot root 4, so search its children.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-focus"><span>3</span><small>try</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>4</span></span><span class="coding-trace-tree-node"><span>5</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>1</span></span><span class="coding-trace-tree-node"><span>2</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Match at node 4"><div class="coding-trace-frame-heading"><span>Match at node 4</span><strong>The subtree rooted at 4 has the same value and child shape.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>3</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node trace-tone-output"><span>4</span><small>match</small></span><span class="coding-trace-tree-node"><span>5</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node trace-tone-output"><span>1</span><small>match</small></span><span class="coding-trace-tree-node trace-tone-output"><span>2</span><small>match</small></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return true"><div class="coding-trace-frame-heading"><span>Return true</span><strong>One complete matching subtree is enough.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>3</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node trace-tone-output"><span>4</span><small>subtree</small></span><span class="coding-trace-tree-node"><span>5</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>1</span></span><span class="coding-trace-tree-node"><span>2</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Scan candidate roots</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Match at node 4</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return true</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Try the full-tree equality test at each candidate node.</p></div><figcaption><strong>Read it this way:</strong> The root 3 does not match subroot root 4, so search its children. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
