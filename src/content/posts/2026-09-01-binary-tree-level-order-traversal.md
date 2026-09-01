---
title: "Binary Tree Level Order Traversal"
description: "Return tree values one level at a time."
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

> Return tree values one level at a time.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:binary-tree-level-order-traversal-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-tree-level-order-traversal-state-title"><p class="visual-kicker">Distance in layers</p><p class="visual-title" id="binary-tree-level-order-traversal-state-title">Binary Tree Level Order Traversal: A queue turns time or steps into visible layers</p><div class="coding-visual coding-visual--bfs" data-coding-visual data-coding-mode="bfs" data-coding-slug="binary-tree-level-order-traversal" role="group" aria-label="Binary Tree Level Order Traversal: queue [3] -&gt; read one layer, then append its children 9 and 20. The queue is ordered by nondecreasing distance from the starting frontier."><div class="coding-visual-example"><span>Concrete trace</span><strong>queue [3] -&gt; read one layer, then append its children 9 and 20</strong></div><div class="coding-visual-sketch coding-visual-sketch--bfs"><div class="coding-sketch-grid"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--seen">0</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--frontier">1</span><span class="coding-sketch-grid-cell">2</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--seen">1</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--frontier">2</span><span class="coding-sketch-grid-cell">3</span><span class="coding-sketch-grid-cell">2</span><span class="coding-sketch-grid-cell">3</span><span class="coding-sketch-grid-cell">4</span></div><p class="coding-sketch-note">each layer is one more step or minute from the starting frontier</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Seed</span><strong>frontier at 0</strong><small>Put every starting position in the queue.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Pop</span><strong>current layer</strong><small>Process only positions at the same distance.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Spread</span><strong>next layer</strong><small>Add each newly reachable neighbor once.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Finish</span><strong>first arrival</strong><small>The first layer reaching a goal is shortest.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The queue is ordered by nondecreasing distance from the starting frontier.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read each queue layer as one minute or one step. Multiple starting points belong in the first layer, which is why multi-source BFS measures the nearest source. For this problem, hold onto the concrete trace: queue [3] -&gt; read one layer, then append its children 9 and 20.</figcaption></figure>

**Pattern:** BFS.

**Simple idea:** The queue contains the next level. Read its current size before adding any
children. That size tells you how many nodes belong to this level.

```python
from __future__ import annotations
from collections import deque
from dataclasses import dataclass

@dataclass(eq=False, slots=True)
class TreeNode:
   val: int
   left: TreeNode | None = None
   right: TreeNode | None = None

def level_order(root: TreeNode | None) -> list[list[int]]:
   if root is None:
      return []

   answer: list[list[int]] = []
   queue = deque([root])

   while queue:
      level = []
      for _ in range(len(queue)):
         node = queue.popleft()
         level.append(node.val)
         if node.left:
            queue.append(node.left)
         if node.right:
            queue.append(node.right)
      answer.append(level)
   return answer
```

**Cost:** $O(n)$ time and $O(w)$ space, where $w$ is the widest level.
