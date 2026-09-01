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
<figure class="learning-figure coding-visual-figure" aria-labelledby="merge-k-sorted-lists-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="merge-k-sorted-lists-state-title">Merge K Sorted Lists: The heap holds one current head per list; pop the smallest and replace it with that list next.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="merge-k-sorted-lists" role="group" aria-label="Merge K Sorted Lists: The heap holds one current head per list; pop the smallest and replace it with that list next."><div class="coding-visual-example"><span>Input and goal</span><strong>Merge many sorted linked lists into one sorted list.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Seed one head per list"><div class="coding-trace-frame-heading"><span>Seed one head per list</span><strong>The heap contains 1 from list A, 1 from B, and 2 from C.</strong></div><div class="coding-trace-heap"><div class="coding-trace-heap-tree"><span class="coding-trace-heap-node is-root">A:1</span><span class="coding-trace-heap-node">B:1</span><span class="coding-trace-heap-node">C:2</span></div></div><div class="coding-trace-meta"><span><b>root</b>A:1</span><span><b>detail</b>one head per list</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Pop and replace"><div class="coding-trace-frame-heading"><span>Pop and replace</span><strong>After taking A:1, insert A:4 while B:1 remains the root.</strong></div><div class="coding-trace-heap"><div class="coding-trace-heap-tree"><span class="coding-trace-heap-node is-root">B:1</span><span class="coding-trace-heap-node">C:2</span><span class="coding-trace-heap-node">A:4</span></div></div><div class="coding-trace-meta"><span><b>root</b>B:1</span><span><b>detail</b>replace from same list</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Finish from the remaining heads"><div class="coding-trace-frame-heading"><span>Finish from the remaining heads</span><strong>After emitting 1,1,2,3,4,4, the remaining heads are 5 and 6.</strong></div><div class="coding-trace-heap"><div class="coding-trace-heap-tree"><span class="coding-trace-heap-node is-root">A:5</span><span class="coding-trace-heap-node">C:6</span></div></div><div class="coding-trace-meta"><span><b>root</b>A:5</span><span><b>detail</b>emit 5, then 6</span><span><b>result</b>1,1,2,3,4,4,5,6</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Seed one head per list</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Pop and replace</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Finish from the remaining heads</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>The heap holds one current head per list; pop the smallest and replace it with that list next.</p></div><figcaption><strong>Read it this way:</strong> The heap contains 1 from list A, 1 from B, and 2 from C. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
