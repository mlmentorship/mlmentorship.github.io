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
<figure class="learning-figure coding-visual-figure" aria-labelledby="maximum-depth-of-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="maximum-depth-of-binary-tree-state-title">Maximum Depth of Binary Tree: A node returns one plus the larger depth returned by its children.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="maximum-depth-of-binary-tree" role="group" aria-label="Maximum Depth of Binary Tree: A node returns one plus the larger depth returned by its children."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the number of nodes on the longest root-to-leaf path.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Solve the leaves"><div class="coding-trace-frame-heading"><span>Solve the leaves</span><strong>Every leaf returns depth 1.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>3</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node trace-tone-state"><span>9</span><small>1</small></span><span class="coding-trace-tree-node"><span>20</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node trace-tone-state"><span>15</span><small>1</small></span><span class="coding-trace-tree-node trace-tone-state"><span>7</span><small>1</small></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Combine child depths"><div class="coding-trace-frame-heading"><span>Combine child depths</span><strong>Node 20 receives 1 and 1, so its depth is 2.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node"><span>3</span></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>9</span></span><span class="coding-trace-tree-node trace-tone-focus"><span>20</span><small>depth 2</small></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>15</span></span><span class="coding-trace-tree-node"><span>7</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return to the root"><div class="coding-trace-frame-heading"><span>Return to the root</span><strong>Root 3 returns 1 + max(1,2) = 3.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-output"><span>3</span><small>depth 3</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>9</span></span><span class="coding-trace-tree-node"><span>20</span></span></div><div class="coding-trace-tree-level" data-level="2"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>15</span></span><span class="coding-trace-tree-node"><span>7</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Solve the leaves</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Combine child depths</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return to the root</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A node returns one plus the larger depth returned by its children.</p></div><figcaption><strong>Read it this way:</strong> Every leaf returns depth 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
