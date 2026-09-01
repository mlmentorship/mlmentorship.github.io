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
<figure class="learning-figure coding-visual-figure" aria-labelledby="reverse-linked-list-state-title"><p class="visual-kicker">Pointers in motion</p><p class="visual-title" id="reverse-linked-list-state-title">Reverse Linked List: Save the next link before redirecting the current one</p><div class="coding-visual coding-visual--linked" data-coding-visual data-coding-mode="linked" data-coding-slug="reverse-linked-list" role="group" aria-label="Reverse Linked List: 1-&gt;2-&gt;3 becomes 1&lt;-2&lt;-3; save 2 before redirecting 1. Every node is still reachable through either the saved suffix or the rebuilt prefix."><div class="coding-visual-example"><span>Concrete trace</span><strong>1-&gt;2-&gt;3 becomes 1&lt;-2&lt;-3; save 2 before redirecting 1</strong></div><div class="coding-visual-sketch coding-visual-sketch--linked"><div class="coding-sketch-path"><span class="coding-sketch-node coding-sketch-node--state">previous</span><span class="coding-sketch-arrow">&larr;</span><span class="coding-sketch-node coding-sketch-node--active">current</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node">saved next</span></div><p class="coding-sketch-note">save the outgoing link before redirecting the current node</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Save</span><strong>next pointer</strong><small>Keep the only route to the unrevised suffix.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Redirect</span><strong>current.next</strong><small>Change one link to the new direction.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Advance</span><strong>previous / current</strong><small>Move the working window by one node.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Return</span><strong>new head</strong><small>The pointer at the boundary becomes the result.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every node is still reachable through either the saved suffix or the rebuilt prefix.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The list is a chain of ownership. First save the outgoing link, then edit the link you own, then advance into the saved suffix. For this problem, hold onto the concrete trace: 1-&gt;2-&gt;3 becomes 1&lt;-2&lt;-3; save 2 before redirecting 1.</figcaption></figure>

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
