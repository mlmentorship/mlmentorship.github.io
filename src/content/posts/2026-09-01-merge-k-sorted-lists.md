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

Combine several sorted linked lists into one sorted chain containing their original nodes exactly once.

## Problem

**Input.** A finite list of linked-list head references, each describing a nondecreasing integer chain or an empty chain.

**Output.** Return the head of one nondecreasing chain containing every node from every input list.

**Rules.** Reuse the original nodes and rewrite their links rather than copying values into replacement nodes.
Input chains must be acyclic and pairwise node-disjoint. Equal values remain separate occurrences.
Leave the head array unchanged. Empty sources contribute nothing; no nodes returns `None`.

**Example.** Chains `1 -> 4`, `1 -> 3`, and `2 -> 6` merge into `1 -> 1 -> 2 -> 3 -> 4 -> 6`.
Both nodes valued 1 must remain in the result.

## Picture

For `A:1->4`, `B:1->3`, keep candidates `(1,0,A)` and `(1,1,B)`.
Emit A's `1`, then replace only A's candidate with `4`.
B's `1` is now smallest.

<!-- visual:merge-k-sorted-lists-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="merge-k-sorted-lists-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="merge-k-sorted-lists-state-title">Emitting one frontier advances only the list that supplied it</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="merge-k-sorted-lists" role="group" tabindex="0" aria-label="Merge K Sorted Lists: A min-heap containing one current head per nonempty list always exposes the globally smallest unmerged node."><div class="coding-visual-example"><span>Input and goal</span><strong>Merge many sorted linked lists into one sorted list.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="seed-current-heads" role="group" aria-label="Seed one head from each list"><div class="coding-trace-frame-heading"><span>Seed one head from each list</span><strong>Push tuples (1,0,A1), (1,1,B1), and (2,2,C2). The list index breaks the value-1 tie, so A1 is the root.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>candidate min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-(1,0,A1)-0">(1,0,A1)</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(1,1,B1)-0">(1,1,B1)</span></li><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(2,2,C2)-0">(2,2,C2)</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>inputLists</b>A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</span><span><b>heapArray</b>[(1,0,A1),(1,1,B1),(2,2,C2)]</span><span><b>tail</b>dummy</span><span><b>outputOrder</b>[]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>input lists</dt><dd><strong class="coding-state-after">A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</strong></dd></div><div class="coding-state-change"><dt>heap array</dt><dd><strong class="coding-state-after">[(1,0,A1),(1,1,B1),(2,2,C2)]</strong></dd></div><div class="coding-state-change"><dt>tail</dt><dd><strong class="coding-state-after">dummy</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="emit-a1" hidden role="group" aria-label="Emit A1 and replace it with A4"><div class="coding-trace-frame-heading"><span>Emit A1 and replace it with A4</span><strong>Pop (1,0,A1), write dummy.next = A1, and set tail = A1. Because A1.next is A4, push (4,0,A4).</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>candidate min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-(1,1,B1)-0">(1,1,B1)</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(2,2,C2)-0">(2,2,C2)</span></li><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(4,0,A4)-0">(4,0,A4)</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>inputLists</b>A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</span><span><b>popped</b>(1,0,A1)</span><span><b>pushed</b>(4,0,A4)</span><span><b>tail</b>A1</span><span><b>outputOrder</b>[A1]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1,0,A1)</strong></dd></div><div class="coding-state-change"><dt>pushed</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(4,0,A4)</strong></dd></div><div class="coding-state-change"><dt>tail</dt><dd><span class="coding-state-before">dummy</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">A1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="emit-b1" hidden role="group" aria-label="Emit B1 and replace it with B3"><div class="coding-trace-frame-heading"><span>Emit B1 and replace it with B3</span><strong>Pop (1,1,B1), overwrite A1.next = B1, and advance tail. Push B1.next as (3,1,B3).</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>candidate min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-(2,2,C2)-0">(2,2,C2)</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(4,0,A4)-0">(4,0,A4)</span></li><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(3,1,B3)-0">(3,1,B3)</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>inputLists</b>A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</span><span><b>popped</b>(1,1,B1)</span><span><b>pushed</b>(3,1,B3)</span><span><b>tail</b>B1</span><span><b>outputOrder</b>[A1,B1]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before">(1,0,A1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1,1,B1)</strong></dd></div><div class="coding-state-change"><dt>pushed</dt><dd><span class="coding-state-before">(4,0,A4)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(3,1,B3)</strong></dd></div><div class="coding-state-change"><dt>tail</dt><dd><span class="coding-state-before">A1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">B1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="emit-c2" hidden role="group" aria-label="Emit C2 and replace it with C6"><div class="coding-trace-frame-heading"><span>Emit C2 and replace it with C6</span><strong>Pop (2,2,C2), link B1.next = C2, then push C2.next as (6,2,C6).</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>candidate min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-(3,1,B3)-0">(3,1,B3)</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(4,0,A4)-0">(4,0,A4)</span></li><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(6,2,C6)-0">(6,2,C6)</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>inputLists</b>A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</span><span><b>popped</b>(2,2,C2)</span><span><b>pushed</b>(6,2,C6)</span><span><b>tail</b>C2</span><span><b>outputOrder</b>[A1,B1,C2]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before">(1,1,B1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2,2,C2)</strong></dd></div><div class="coding-state-change"><dt>pushed</dt><dd><span class="coding-state-before">(3,1,B3)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(6,2,C6)</strong></dd></div><div class="coding-state-change"><dt>tail</dt><dd><span class="coding-state-before">B1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">C2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="emit-b3" hidden role="group" aria-label="Emit B3 and replace it with B4"><div class="coding-trace-frame-heading"><span>Emit B3 and replace it with B4</span><strong>Pop (3,1,B3), link C2.next = B3, then push (4,1,B4). A4 wins the value-4 tie because list index 0 is smaller than 1.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>candidate min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-(4,0,A4)-0">(4,0,A4)</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(6,2,C6)-0">(6,2,C6)</span></li><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(4,1,B4)-0">(4,1,B4)</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>inputLists</b>A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</span><span><b>popped</b>(3,1,B3)</span><span><b>pushed</b>(4,1,B4)</span><span><b>tail</b>B3</span><span><b>outputOrder</b>[A1,B1,C2,B3]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before">(2,2,C2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(3,1,B3)</strong></dd></div><div class="coding-state-change"><dt>pushed</dt><dd><span class="coding-state-before">(6,2,C6)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(4,1,B4)</strong></dd></div><div class="coding-state-change"><dt>tail</dt><dd><span class="coding-state-before">C2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">B3</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="emit-a4" hidden role="group" aria-label="Emit A4 and replace it with A5"><div class="coding-trace-frame-heading"><span>Emit A4 and replace it with A5</span><strong>Pop (4,0,A4), link B3.next = A4, and push A4.next as (5,0,A5).</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>candidate min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-(4,1,B4)-0">(4,1,B4)</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(6,2,C6)-0">(6,2,C6)</span></li><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(5,0,A5)-0">(5,0,A5)</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>inputLists</b>A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</span><span><b>popped</b>(4,0,A4)</span><span><b>pushed</b>(5,0,A5)</span><span><b>tail</b>A4</span><span><b>outputOrder</b>[A1,B1,C2,B3,A4]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before">(3,1,B3)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(4,0,A4)</strong></dd></div><div class="coding-state-change"><dt>pushed</dt><dd><span class="coding-state-before">(4,1,B4)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(5,0,A5)</strong></dd></div><div class="coding-state-change"><dt>tail</dt><dd><span class="coding-state-before">B3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">A4</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="emit-b4" hidden role="group" aria-label="Emit B4 without replacement"><div class="coding-trace-frame-heading"><span>Emit B4 without replacement</span><strong>Pop (4,1,B4) and link A4.next = B4. B4.next is null, so the if branch skips heappush.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>candidate min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-(5,0,A5)-0">(5,0,A5)</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-(6,2,C6)-0">(6,2,C6)</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>inputLists</b>A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</span><span><b>popped</b>(4,1,B4)</span><span><b>pushed</b>none: node.next is null</span><span><b>tail</b>B4</span><span><b>outputOrder</b>[A1,B1,C2,B3,A4,B4]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before">(4,0,A4)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(4,1,B4)</strong></dd></div><div class="coding-state-change"><dt>pushed</dt><dd><span class="coding-state-before">(5,0,A5)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">none: node.next is null</strong></dd></div><div class="coding-state-change"><dt>tail</dt><dd><span class="coding-state-before">A4</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">B4</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="7" data-frame-key="emit-a5" hidden role="group" aria-label="Emit A5 without replacement"><div class="coding-trace-frame-heading"><span>Emit A5 without replacement</span><strong>Pop (5,0,A5), link B4.next = A5, and skip the push because A5.next is null.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>candidate min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-(6,2,C6)-0">(6,2,C6)</span></li></ul></section></div><div class="coding-trace-meta"><span><b>inputLists</b>A:1-&gt;4-&gt;5; B:1-&gt;3-&gt;4; C:2-&gt;6</span><span><b>popped</b>(5,0,A5)</span><span><b>pushed</b>none: node.next is null</span><span><b>tail</b>A5</span><span><b>outputOrder</b>[A1,B1,C2,B3,A4,B4,A5]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before">(4,1,B4)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(5,0,A5)</strong></dd></div><div class="coding-state-change"><dt>tail</dt><dd><span class="coding-state-before">B4</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">A5</strong></dd></div><div class="coding-state-change"><dt>output order</dt><dd><span class="coding-state-before">[A1,B1,C2,B3,A4,B4]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[A1,B1,C2,B3,A4,B4,A5]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="8" data-frame-key="emit-c6" hidden role="group" aria-label="Emit C6 and empty the heap"><div class="coding-trace-frame-heading"><span>Emit C6 and empty the heap</span><strong>Pop (6,2,C6), link A5.next = C6, and skip the push because C6.next is null. The heap is now empty.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-A1"><span>A1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-A1-B1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-B1"><span>B1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-B1-C2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-C2"><span>C2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-C2-B3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-B3"><span>B3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-B3-A4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-A4"><span>A4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-A4-B4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-B4"><span>B4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-B4-A5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-A5"><span>A5</span></span><span class="coding-trace-link-arrow" data-motion-key="link-A5-C6">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-C6"><span>C6</span><small data-motion-key="pointer-tail">tail</small></span></div></div><div class="coding-trace-meta"><span><b>heapState</b>[]</span><span><b>popped</b>(6,2,C6)</span><span><b>pushed</b>none: node.next is null</span><span><b>values</b>1,1,2,3,4,4,5,6</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>heap state</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[]</strong></dd></div><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before">(5,0,A5)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(6,2,C6)</strong></dd></div><div class="coding-state-change"><dt>values</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1,1,2,3,4,4,5,6</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="9" data-frame-key="return-merged-list" hidden role="group" aria-label="Return the merged head"><div class="coding-trace-frame-heading"><span>Return the merged head</span><strong>The while loop stops when the heap empties. Return dummy.next, which is A1 and heads values 1-&gt;1-&gt;2-&gt;3-&gt;4-&gt;4-&gt;5-&gt;6.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-A1"><span>A1</span><small data-motion-key="pointer-dummy.next">dummy.next</small></span><span class="coding-trace-link-arrow" data-motion-key="link-A1-B1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-B1"><span>B1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-B1-C2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-C2"><span>C2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-C2-B3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-B3"><span>B3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-B3-A4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-A4"><span>A4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-A4-B4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-B4"><span>B4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-B4-A5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-A5"><span>A5</span></span><span class="coding-trace-link-arrow" data-motion-key="link-A5-C6">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-C6"><span>C6</span></span></div></div><div class="coding-trace-meta"><span><b>heapState</b>[]</span><span><b>nodeOrder</b>A1-&gt;B1-&gt;C2-&gt;B3-&gt;A4-&gt;B4-&gt;A5-&gt;C6</span><span><b>result</b>1-&gt;1-&gt;2-&gt;3-&gt;4-&gt;4-&gt;5-&gt;6</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>node order</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">A1-&gt;B1-&gt;C2-&gt;B3-&gt;A4-&gt;B4-&gt;A5-&gt;C6</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1-&gt;1-&gt;2-&gt;3-&gt;4-&gt;4-&gt;5-&gt;6</strong></dd></div><div class="coding-state-change"><dt>popped</dt><dd><span class="coding-state-before">(6,2,C6)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 10</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Seed one head from each list</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Emit A1 and replace it with A4</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Emit B1 and replace it with B3</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Emit C2 and replace it with C6</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Emit B3 and replace it with B4</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Emit A4 and replace it with A5</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Emit B4 without replacement</strong></button><button type="button" data-coding-frame-button="7"><span>8</span><strong>Emit A5 without replacement</strong></button><button type="button" data-coding-frame-button="8"><span>9</span><strong>Emit C6 and empty the heap</strong></button><button type="button" data-coding-frame-button="9"><span>10</span><strong>Return the merged head</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when k individually sorted streams must be merged and only their current heads can be candidates for the next global minimum.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">The heap contains exactly the first unmerged node of each nonexhausted list, and the tail follows all nodes already popped in nondecreasing tuple order.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For sorted sources, keep only one frontier item per source and replace it from the same source after extraction. The pattern transfers to external merge, log aggregation, and k-way iterators.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Push tuples (1,0,A1), (1,1,B1), and (2,2,C2). The list index breaks the value-1 tie, so A1 is the root. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="merge-k-sorted-lists-state"><strong>Check:</strong> Why does emitting A1 add A4 to the heap but leave the B and C frontiers unchanged?</span></figcaption></figure>

Every hidden source node is at least its exposed head.
The heap minimum is therefore globally safe.
The source index resolves equal values without comparing node objects.

## Code

This local definition supplies the platform node type before the unchanged reference.

```python
class ListNode:
   def __init__(self, val: int = 0, next_node: "ListNode | None" = None):
      self.val = val
      self.next = next_node
```

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

**Cost:** $O(k+n\log(k+1))$ scalar operations and $O(k)$ heap space for `k` sources and `n` nodes.
The reference allocates only one dummy node.

The platform supplies `ListNode` with `val` and `next`; this snippet assumes that definition.

## Change the Requirement

### Preserve All Input Links

**New task:** Return independent output nodes.

**Picture and change:** Copy the emitted value instead of reusing its node.

**Code:** Use `heapq` and `ListNode` above.

```python
def merge_k_lists_copy(lists: list[ListNode | None]) -> ListNode | None:
   heap = [(node.val, index, node) for index, node in enumerate(lists) if node]
   heapq.heapify(heap)
   dummy = tail = ListNode()
   while heap:
      value, index, node = heapq.heappop(heap)
      tail.next = ListNode(value)
      tail = tail.next
      if node.next:
         heapq.heappush(heap, (node.next.val, index, node.next))
   return dummy.next
```

**Check:** `A:1->4`, `B:1->3` yield fresh `1->1->3->4` nodes; both inputs retain their links.

**Use when:** Output ownership must be independent; allocation adds $O(n)$.

### Consume Sorted Iterables

**New task:** Merge values without linked nodes.

**Picture and change:** Each heap entry owns one iterator and its current value.

**Code:** The collection of sources must be finite.

```python
def merge_sorted_iterables(sources):
   heap = []
   for index, source in enumerate(sources):
      iterator = iter(source)
      try:
         value = next(iterator)
      except StopIteration:
         continue
      heapq.heappush(heap, (value, index, iterator))
   while heap:
      value, index, iterator = heapq.heappop(heap)
      yield value
      try:
         following = next(iterator)
      except StopIteration:
         continue
      heapq.heappush(heap, (following, index, iterator))
```

**Check:** `list(merge_sorted_iterables([[1,4],[1,3],[]]))` returns `[1,1,3,4]`.

**Use when:** Sources advance lazily; heap state remains $O(k)$.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why require disjoint nodes for the relinking reference?

<details>
<summary>Reasoned answer</summary>

If two heads reference the same node, that object can enter the heap twice.
Appending it twice may set its `next` pointer to itself, creating a cycle.
Equal values are safe; repeated identities are not.
The copy variant avoids output-link mutation but still assumes finite acyclic source traversals.

Infinite value streams need a separate liveness contract.
An infinite sequence of equal values from one source can indefinitely postpone equal values from another source.
Sorted output does not imply fairness among sources.

</details>

</div>
