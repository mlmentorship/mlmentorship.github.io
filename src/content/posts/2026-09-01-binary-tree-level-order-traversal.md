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
<figure class="learning-figure coding-visual-figure" aria-labelledby="binary-tree-level-order-traversal-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="binary-tree-level-order-traversal-state-title">Binary Tree Level Order Traversal: Read exactly the queue length that existed before adding child nodes.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="binary-tree-level-order-traversal" role="group" aria-label="Binary Tree Level Order Traversal: Read exactly the queue length that existed before adding child nodes."><div class="coding-visual-example"><span>Input and goal</span><strong>Return tree values one level at a time.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Queue the root"><div class="coding-trace-frame-heading"><span>Queue the root</span><strong>The first layer contains only 3.</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">3</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">9</span><span class="coding-trace-grid-cell">20</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">2</span><span class="coding-trace-grid-cell">15</span><span class="coding-trace-grid-cell">7</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-queue-item">3</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Read one level"><div class="coding-trace-frame-heading"><span>Read one level</span><strong>Pop 3, then append 9 and 20 for the next layer.</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">3</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">9</span><span class="coding-trace-grid-cell">20</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">2</span><span class="coding-trace-grid-cell">15</span><span class="coding-trace-grid-cell">7</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-queue-item">9</span><span class="coding-trace-queue-item">20</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Continue by layer"><div class="coding-trace-frame-heading"><span>Continue by layer</span><strong>The queue boundary gives [[3],[9,20],[15,7]].</strong></div><div class="coding-trace-queue-grid"><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">0</span><span class="coding-trace-grid-cell">3</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">1</span><span class="coding-trace-grid-cell">9</span><span class="coding-trace-grid-cell">20</span></div><div class="coding-trace-queue-grid-row"><span class="coding-trace-label">2</span><span class="coding-trace-grid-cell">15</span><span class="coding-trace-grid-cell">7</span></div></div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span><span class="coding-trace-empty">empty</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Queue the root</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Read one level</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Continue by layer</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Read exactly the queue length that existed before adding child nodes.</p></div><figcaption><strong>Read it this way:</strong> The first layer contains only 3. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
