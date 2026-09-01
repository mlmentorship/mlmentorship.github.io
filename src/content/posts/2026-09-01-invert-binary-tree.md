---
title: "Invert Binary Tree"
description: "Swap the left and right children at every node."
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

> Swap the left and right children at every node.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:invert-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="invert-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="invert-binary-tree-state-title">Invert Binary Tree: Swap the two child links at every node.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="invert-binary-tree" role="group" tabindex="0" aria-label="Invert Binary Tree: Swap the two child links at every node."><div class="coding-visual-example"><span>Input and goal</span><strong>Swap the left and right children at every node.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Original children"><div class="coding-trace-frame-heading"><span>Original children</span><strong>Node 2 points left to 1 and right to 3.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 144" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-2-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">2</text><text class="coding-trace-node-state" x="320" y="58">current</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">1</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">3</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Swap at the root"><div class="coding-trace-frame-heading"><span>Swap at the root</span><strong>The root now points left to 3 and right to 1.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 144" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><g class="coding-trace-tree-node trace-tone-focus" data-motion-key="tree-node-2-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">2</text><text class="coding-trace-node-state" x="320" y="58">swapped</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">3</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">1</text></g></svg></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the inverted tree"><div class="coding-trace-frame-heading"><span>Return the inverted tree</span><strong>The same swap happens recursively below every node.</strong></div><div class="coding-trace-tree"><svg viewBox="0 0 640 144" role="img" aria-label="Binary tree with parent-child edges and call state"><line class="coding-trace-edge-line" x1="320" y1="28" x2="160" y2="100" /><line class="coding-trace-edge-line" x1="320" y1="28" x2="480" y2="100" /><g class="coding-trace-tree-node trace-tone-output" data-motion-key="tree-node-2-0"><circle cx="320" cy="28" r="18" /><text x="320" y="32">2</text><text class="coding-trace-node-state" x="320" y="58">done</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-3-0"><circle cx="160" cy="100" r="18" /><text x="160" y="104">3</text></g><g class="coding-trace-tree-node" data-motion-key="tree-node-1-0"><circle cx="480" cy="100" r="18" /><text x="480" y="104">1</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>inverted</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Original children</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Swap at the root</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the inverted tree</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Swap the two child links at every node.</p></div><figcaption><strong>Read it this way:</strong> Node 2 points left to 1 and right to 3. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Tree DFS.

**Simple idea:** Invert both child trees, then place the old right result on the left and the
old left result on the right.

```python
def invert_tree(root: TreeNode | None) -> TreeNode | None:
   if root is None:
      return None
   root.left, root.right = invert_tree(root.right), invert_tree(root.left)
   return root
```

**Cost:** $O(n)$ time and $O(h)$ call-stack space.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
