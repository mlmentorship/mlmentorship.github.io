---
title: "Reorder List"
description: "Change `1, 2, 3, 4, 5` into `1, 5, 2, 4, 3`."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Mixed"
priority: "Core"
aliases: []
prerequisites: []
---

> Change `1, 2, 3, 4, 5` into `1, 5, 2, 4, 3`.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:reorder-list-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="reorder-list-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="reorder-list-state-title">Reorder List: Find the middle, reverse the second half, then interleave the two lists.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="reorder-list" role="group" aria-label="Reorder List: Find the middle, reverse the second half, then interleave the two lists."><div class="coding-visual-example"><span>Input and goal</span><strong>Change `1, 2, 3, 4, 5` into `1, 5, 2, 4, 3`.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Split at the middle"><div class="coding-trace-frame-heading"><span>Split at the middle</span><strong>Slow and fast leave first half 1,2,3 and second half 4,5.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node"><span>1</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>2</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>3</span><small>split</small></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>4</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>detail</b>first: 1-&gt;2-&gt;3; second: 4-&gt;5</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Reverse the second half"><div class="coding-trace-frame-heading"><span>Reverse the second half</span><strong>The second list becomes 5-&gt;4.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node"><span>1</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>2</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>3</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node trace-tone-focus"><span>5</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>4</span></span></div></div><div class="coding-trace-meta"><span><b>detail</b>second: 5-&gt;4</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Interleave"><div class="coding-trace-frame-heading"><span>Interleave</span><strong>Take one node from each half: 1,5,2,4,3.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node trace-tone-output"><span>1</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node trace-tone-output"><span>5</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node trace-tone-output"><span>2</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node trace-tone-output"><span>4</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node trace-tone-output"><span>3</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>1-&gt;5-&gt;2-&gt;4-&gt;3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Split at the middle</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Reverse the second half</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Interleave</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Find the middle, reverse the second half, then interleave the two lists.</p></div><figcaption><strong>Read it this way:</strong> Slow and fast leave first half 1,2,3 and second half 4,5. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Find middle, reverse second half, then merge.

**Simple idea:** This problem combines three linked-list moves. Slow and fast pointers find
the middle. Reverse the second half. Alternate nodes from the two halves.

```python
def reorder_list(head: ListNode | None) -> None:
   if head is None or head.next is None:
      return

   slow = head
   fast = head
   while fast.next and fast.next.next:
      slow = slow.next
      fast = fast.next.next

   second = slow.next
   slow.next = None
   previous = None
   while second:
      next_node = second.next
      second.next = previous
      previous = second
      second = next_node

   first = head
   second = previous
   while second:
      first_next = first.next
      second_next = second.next
      first.next = second
      second.next = first_next
      first = first_next
      second = second_next
```

**Cost:** $O(n)$ time and $O(1)$ space.

The platform supplies `ListNode` with `val` and `next`; this snippet assumes that definition.
