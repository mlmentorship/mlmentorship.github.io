---
title: "Linked List Cycle"
description: "Check whether a linked list contains a cycle."
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

> Check whether a linked list contains a cycle.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:linked-list-cycle-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="linked-list-cycle-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="linked-list-cycle-state-title">Linked List Cycle: A one-step pointer and a two-step pointer must meet inside a cycle.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="linked-list-cycle" role="group" aria-label="Linked List Cycle: A one-step pointer and a two-step pointer must meet inside a cycle."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether a linked list contains a cycle.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Move at different speeds"><div class="coding-trace-frame-heading"><span>Move at different speeds</span><strong>After one move, slow is at 2 and fast is at 3.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node"><span>1</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>2</span><small>slow</small></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>3</span><small>fast</small></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>4</span></span></div><p class="coding-trace-inline-note">1 -&gt; 2 · 2 -&gt; 3 · 3 -&gt; 4 · 4 -&gt; 2</p></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Enter the loop"><div class="coding-trace-frame-heading"><span>Enter the loop</span><strong>After the next move, slow is at 3 and fast has wrapped to 2.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node"><span>2</span><small>fast</small></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>3</span><small>slow</small></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>4</span></span></div><p class="coding-trace-inline-note">2 -&gt; 3 · 3 -&gt; 4 · 4 -&gt; 2</p></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Meet"><div class="coding-trace-frame-heading"><span>Meet</span><strong>On the next move both pointers reach 4, proving a cycle exists.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node trace-tone-output"><span>4</span><small>slow + fast</small></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>2</span></span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-linked-node"><span>3</span></span></div><p class="coding-trace-inline-note">4 -&gt; 2 · 2 -&gt; 3 · 3 -&gt; 4</p></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Move at different speeds</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Enter the loop</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Meet</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A one-step pointer and a two-step pointer must meet inside a cycle.</p></div><figcaption><strong>Read it this way:</strong> After one move, slow is at 2 and fast is at 3. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Slow and fast pointers.

**Simple idea:** One pointer moves one step and the other moves two. Inside a cycle, the fast
pointer must catch the slow pointer. Without a cycle, the fast pointer reaches the end.

```python
def has_cycle(head: ListNode | None) -> bool:
   slow = head
   fast = head

   while fast and fast.next:
      slow = slow.next
      fast = fast.next.next
      if slow is fast:
         return True
   return False
```

**Cost:** $O(n)$ time and $O(1)$ space.

The platform supplies `ListNode` with `val` and `next`; this snippet assumes that definition.
