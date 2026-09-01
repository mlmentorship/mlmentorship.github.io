---
title: "Reverse Linked List"
description: "Reverse all links in a singly linked list."
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

> Reverse all links in a singly linked list.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:reverse-linked-list-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="reverse-linked-list-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="reverse-linked-list-state-title">Reverse Linked List: Save the outgoing link, reverse the current link, then advance.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="reverse-linked-list" role="group" tabindex="0" aria-label="Reverse Linked List: Save the outgoing link, reverse the current link, then advance."><div class="coding-visual-example"><span>Input and goal</span><strong>Reverse all links in a singly linked list.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Save next"><div class="coding-trace-frame-heading"><span>Save next</span><strong>Before changing 1.next, save the route to node 2.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span><small data-motion-key="pointer-current">current</small></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span><small data-motion-key="pointer-next">next</small></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span></span></div><p class="coding-trace-inline-note">1 -&gt; 2 · 2 -&gt; 3</p></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Reverse one link"><div class="coding-trace-frame-heading"><span>Reverse one link</span><strong>Point 1 back to previous, then advance current to saved node 2.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span><small data-motion-key="pointer-previous">previous</small></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span><small data-motion-key="pointer-current">current</small></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span><small data-motion-key="pointer-next">next</small></span></div><p class="coding-trace-inline-note">2 -&gt; 3 · 1 -&gt; null</p></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return new head"><div class="coding-trace-frame-heading"><span>Return new head</span><strong>After all links reverse, previous points at 3.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node trace-tone-output" data-motion-key="node-3"><span>3</span><small data-motion-key="pointer-head">head</small></span><span class="coding-trace-link-arrow" data-motion-key="link-3-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span></div><p class="coding-trace-inline-note">3 -&gt; 2 · 2 -&gt; 1</p></div><div class="coding-trace-meta"><span><b>result</b>3 -&gt; 2 -&gt; 1</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Save next</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Reverse one link</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return new head</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Save the outgoing link, reverse the current link, then advance.</p></div><figcaption><strong>Read it this way:</strong> Before changing 1.next, save the route to node 2. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Previous, current, and next pointers.

**Simple idea:** Save the next node. Point the current node backward. Move both working
pointers forward.

```python
def reverse_list(head: ListNode | None) -> ListNode | None:
   previous = None

   while head:
      next_node = head.next
      head.next = previous
      previous = head
      head = next_node

   return previous
```

**Cost:** $O(n)$ time and $O(1)$ space.

The platform supplies `ListNode` with `val` and `next`; this snippet assumes that definition.
