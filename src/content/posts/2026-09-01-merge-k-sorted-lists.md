---
title: "Merge K Sorted Lists"
description: "Merge many sorted linked lists into one sorted list."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

> Merge many sorted linked lists into one sorted list.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:merge-k-sorted-lists-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="merge-k-sorted-lists-state-title"><p class="visual-kicker">A frontier ordered by value</p><p class="visual-title" id="merge-k-sorted-lists-state-title">Merge K Sorted Lists: Keep the candidates that can still win</p><div class="coding-visual coding-visual--heap" data-coding-visual data-coding-mode="heap" data-coding-slug="merge-k-sorted-lists" role="group" aria-label="Merge K Sorted Lists: heap holds one head per list; pop the smallest and add that list&#39;s next node. The heap root is the next candidate whose priority is safe to process."><div class="coding-visual-example"><span>Concrete trace</span><strong>heap holds one head per list; pop the smallest and add that list&#39;s next node</strong></div><div class="coding-visual-sketch coding-visual-sketch--heap"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">root: next best</span><div class="coding-sketch-branch"><span class="coding-sketch-node">candidate</span><span class="coding-sketch-node">candidate</span></div></div><p class="coding-sketch-note">the root is exposed while the rest stays as a frontier</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Offer</span><strong>candidate set</strong><small>Put a new value into the frontier.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expose</span><strong>root = next best</strong><small>The heap makes the smallest or largest current item visible.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Trim</span><strong>keep k</strong><small>Discard a candidate that cannot enter the answer.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Advance</span><strong>next candidate</strong><small>Replace the used item and continue the stream.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The heap root is the next candidate whose priority is safe to process.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The heap is not a sorted list. It exposes only the next useful item, while preserving enough frontier state to continue without sorting everything. For this problem, hold onto the concrete trace: heap holds one head per list; pop the smallest and add that list&#39;s next node.</figcaption></figure>

**Pattern:** Heap with one current node from each list.

**Simple idea:** The next result node must be the smallest current head. The heap finds that
node. After removing it, add the next node from the same list.

```python
import heapq

def merge_k_lists(lists: list[ListNode | None]) -> ListNode | None:
   heap: list[tuple[int, int, ListNode]] = []
   for index, node in enumerate(lists):
      if node:
         heapq.heappush(heap, (node.val, index, node))

   dummy = ListNode(0)
   tail = dummy
   while heap:
      _, index, node = heapq.heappop(heap)
      tail.next = node
      tail = node
      if node.next:
         heapq.heappush(heap, (node.next.val, index, node.next))
   return dummy.next
```

**Cost:** $O(n\log k)$ time and $O(k)$ space, where $n$ is the total node count.

The platform supplies `ListNode` with `val` and `next`; this snippet assumes that definition.
