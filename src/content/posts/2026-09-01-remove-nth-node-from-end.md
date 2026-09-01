---
title: "Remove Nth Node From End"
description: "Remove the `n`th node counted from the end."
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

> Remove the `n`th node counted from the end.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:remove-nth-node-from-end-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="remove-nth-node-from-end-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="remove-nth-node-from-end-state-title">Remove Nth Node From End: A fixed pointer gap makes the left pointer stop just before the node to remove.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="remove-nth-node-from-end" role="group" tabindex="0" aria-label="Remove Nth Node From End: A fixed pointer gap makes the left pointer stop just before the node to remove."><div class="coding-visual-example"><span>Input and goal</span><strong>Remove the `n`th node counted from the end.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Create a gap"><div class="coding-trace-frame-heading"><span>Create a gap</span><strong>Move right two nodes ahead of left for n=2.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-dummy"><span>dummy</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-dummy-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span><small data-motion-key="pointer-right">right</small></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span></span></div><p class="coding-trace-inline-note">dummy -&gt; 1 -&gt; 2 -&gt; 3 -&gt; 4 -&gt; 5</p></div><div class="coding-trace-meta"><span><b>detail</b>gap = 2</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Walk together"><div class="coding-trace-frame-heading"><span>Walk together</span><strong>When right reaches 5, left is at node 3.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span><small data-motion-key="pointer-right">right</small></span></div><p class="coding-trace-inline-note">3 -&gt; 4 -&gt; 5</p></div><div class="coding-trace-meta"><span><b>detail</b>left.next is node 4</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Skip the target"><div class="coding-trace-frame-heading"><span>Skip the target</span><strong>Redirect 3.next around node 4.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node trace-tone-focus" data-motion-key="node-3"><span>3</span><small data-motion-key="pointer-link changed">link changed</small></span><span class="coding-trace-link-arrow" data-motion-key="link-3-5">&rarr;</span><span class="coding-trace-linked-node trace-tone-output" data-motion-key="node-5"><span>5</span></span></div><p class="coding-trace-inline-note">1 -&gt; 2 -&gt; 3 -&gt; 5</p></div><div class="coding-trace-meta"><span><b>result</b>[1,2,3,5]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Create a gap</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Walk together</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Skip the target</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A fixed pointer gap makes the left pointer stop just before the node to remove.</p></div><figcaption><strong>Read it this way:</strong> Move right two nodes ahead of left for n=2. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Two pointers with a fixed gap.

**Simple idea:** Move `right` ahead by `n` nodes. Then move both pointers together. When
`right` reaches the end, `left` is just before the node to remove. A dummy node handles
removing the head without a special case.

```python
def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
   dummy = ListNode(0, head)
   left = dummy
   right = dummy

   for _ in range(n):
      if right.next is None:
         return head
      right = right.next

   while right.next:
      left = left.next
      right = right.next

   if left.next:
      left.next = left.next.next
   return dummy.next
```

**Cost:** $O(n)$ time and $O(1)$ space.

The platform supplies `ListNode` with `val` and `next`; this snippet assumes that definition.
