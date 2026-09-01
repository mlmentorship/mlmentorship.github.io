---
title: "Merge Two Sorted Lists"
description: "Merge two sorted linked lists into one sorted list."
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

> Merge two sorted linked lists into one sorted list.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:merge-two-sorted-lists-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="merge-two-sorted-lists-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="merge-two-sorted-lists-state-title">Merge Two Sorted Lists: Attach the smaller current head and advance only that list.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="merge-two-sorted-lists" role="group" tabindex="0" aria-label="Merge Two Sorted Lists: Attach the smaller current head and advance only that list."><div class="coding-visual-example"><span>Input and goal</span><strong>Merge two sorted linked lists into one sorted list.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Compare two heads"><div class="coding-trace-frame-heading"><span>Compare two heads</span><strong>Heads 1 and 1 tie; attach one and advance its list.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-A:1"><span>A:1</span><small data-motion-key="pointer-head A">head A</small></span><span class="coding-trace-link-arrow" data-motion-key="link-A:1-A:2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-A:2"><span>A:2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-A:2-A:4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-A:4"><span>A:4</span></span></div><div class="coding-trace-linked-row"><span class="coding-trace-label">second</span><span class="coding-trace-linked-node" data-motion-key="node-B:1"><span>B:1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-B:1-B:3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-B:3"><span>B:3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-B:3-B:4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-B:4"><span>B:4</span></span></div></div><div class="coding-trace-meta"><span><b>detail</b>take 1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Continue the merge"><div class="coding-trace-frame-heading"><span>Continue the merge</span><strong>Compare the next heads and attach 2, then 3.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node trace-tone-focus" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node trace-tone-focus" data-motion-key="node-3"><span>3</span></span></div></div><div class="coding-trace-meta"><span><b>detail</b>tail always points at last result node</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Append the remainder"><div class="coding-trace-frame-heading"><span>Append the remainder</span><strong>When one list ends, attach the other suffix unchanged.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>sorted merged list</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Compare two heads</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Continue the merge</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Append the remainder</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Attach the smaller current head and advance only that list.</p></div><figcaption><strong>Read it this way:</strong> Heads 1 and 1 tie; attach one and advance its list. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Two pointers plus a dummy start node.

**Simple idea:** Attach the smaller current node and move only that list. The dummy node
removes the special case for choosing the first result node.

```python
def merge_two_lists(first: ListNode | None, second: ListNode | None) -> ListNode | None:
   dummy = ListNode(0)
   tail = dummy

   while first and second:
      if first.val <= second.val:
         tail.next, first = first, first.next
      else:
         tail.next, second = second, second.next
      tail = tail.next

   tail.next = first or second
   return dummy.next
```

**Cost:** $O(m + n)$ time and $O(1)$ extra space.

The platform supplies `ListNode` with `val` and `next`; this snippet assumes that definition.
