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

Rearrange a linked list into alternating nodes from its beginning and end without replacing its nodes.

## Problem

**Input.** The head of a finite acyclic integer linked list.

**Output.** Rewire the same nodes in place into first, last, second, second-last order, continuing inward; return `None`.

**Rules.** Use every original node exactly once, retain its value, and end the reordered list with `None`.
Do not create a replacement output chain or merely exchange node values.
Values may repeat and need not be sorted. Empty and singleton lists remain unchanged.

**Example.** `1 -> 2 -> 3 -> 4 -> 5` becomes `1 -> 5 -> 2 -> 4 -> 3`.
The four-node list `1 -> 2 -> 3 -> 4` becomes `1 -> 4 -> 2 -> 3`.

## Picture

For `1->2->3->4->5`, the first half is `1->2->3`; the detached second half is `4->5`.
Reverse it to `5->4`, then alternate to obtain `1->5->2->4->3`.

<!-- visual:reorder-list-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="reorder-list-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="reorder-list-state-title">Reorder List: Find the first-half tail, reverse the detached second half, then splice one reversed node after each first-half node.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="reorder-list" role="group" tabindex="0" aria-label="Reorder List: Find the first-half tail, reverse the detached second half, then splice one reversed node after each first-half node."><div class="coding-visual-example"><span>Input and goal</span><strong>Change `1, 2, 3, 4, 5` into `1, 5, 2, 4, 3`.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize-middle-scan" role="group" aria-label="Initialize middle pointers"><div class="coding-trace-frame-heading"><span>Initialize middle pointers</span><strong>For 1-&gt;2-&gt;3-&gt;4-&gt;5, set slow = 1 and fast = 1. Both fast.next and fast.next.next exist.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-1"><span>1</span><small data-motion-key="pointer-slow / fast">slow / fast</small></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-3"><span>3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>guard</b>fast.next=2 and fast.next.next=3</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>guard</dt><dd><strong class="coding-state-after">fast.next=2 and fast.next.next=3</strong></dd></div><div class="coding-state-change"><dt>node 0</dt><dd><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>node 1</dt><dd><strong class="coding-state-after">2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="middle-round-1" hidden role="group" aria-label="Advance middle pointers once"><div class="coding-trace-frame-heading"><span>Advance middle pointers once</span><strong>Move slow one link 1-&gt;2 and fast two links 1-&gt;2-&gt;3. The loop guard still passes at fast = 3.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-2"><span>2</span><small data-motion-key="pointer-slow">slow</small></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-3"><span>3</span><small data-motion-key="pointer-fast">fast</small></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>movement</b>slow: 1-&gt;2; fast: 1-&gt;2-&gt;3</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>movement</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">slow: 1-&gt;2; fast: 1-&gt;2-&gt;3</strong></dd></div><div class="coding-state-change"><dt>guard</dt><dd><span class="coding-state-before">fast.next=2 and fast.next.next=3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="middle-round-2" hidden role="group" aria-label="Advance middle pointers twice"><div class="coding-trace-frame-heading"><span>Advance middle pointers twice</span><strong>Move slow 2-&gt;3 and fast 3-&gt;4-&gt;5. At fast = 5, fast.next is null, so the middle loop stops.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-3"><span>3</span><small data-motion-key="pointer-slow">slow</small></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span><small data-motion-key="pointer-fast">fast</small></span></div></div><div class="coding-trace-meta"><span><b>movement</b>slow: 2-&gt;3; fast: 3-&gt;4-&gt;5</span><span><b>guard</b>fast.next is null</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>movement</dt><dd><span class="coding-state-before">slow: 1-&gt;2; fast: 1-&gt;2-&gt;3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">slow: 2-&gt;3; fast: 3-&gt;4-&gt;5</strong></dd></div><div class="coding-state-change"><dt>guard</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">fast.next is null</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="split-halves" hidden role="group" aria-label="Detach the second half"><div class="coding-trace-frame-heading"><span>Detach the second half</span><strong>Save second = slow.next = 4, then write slow.next = null. The independent halves are 1-&gt;2-&gt;3 and 4-&gt;5.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span><small data-motion-key="pointer-second">second</small></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>firstHalf</b>1-&gt;2-&gt;3-&gt;null</span><span><b>secondHalf</b>4-&gt;5-&gt;null</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>first half</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1-&gt;2-&gt;3-&gt;null</strong></dd></div><div class="coding-state-change"><dt>second half</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4-&gt;5-&gt;null</strong></dd></div><div class="coding-state-change"><dt>movement</dt><dd><span class="coding-state-before">slow: 2-&gt;3; fast: 3-&gt;4-&gt;5</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="reverse-node-4" hidden role="group" aria-label="Reverse second-half node 4"><div class="coding-trace-frame-heading"><span>Reverse second-half node 4</span><strong>With previous = null and second = 4, save next_node = 5, write 4.next = null, then advance previous = 4 and second = 5.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span><small data-motion-key="pointer-previous">previous</small></span></div><div class="coding-trace-linked-row"><span class="coding-trace-label">second</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>firstHalf</b>1-&gt;2-&gt;3-&gt;null</span><span><b>reversed</b>4-&gt;null</span><span><b>savedNext</b>5</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>reversed</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4-&gt;null</strong></dd></div><div class="coding-state-change"><dt>saved next</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5</strong></dd></div><div class="coding-state-change"><dt>second half</dt><dd><span class="coding-state-before">4-&gt;5-&gt;null</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="reverse-node-5" hidden role="group" aria-label="Reverse second-half node 5"><div class="coding-trace-frame-heading"><span>Reverse second-half node 5</span><strong>Save next_node = null, write 5.next = 4, then advance previous = 5 and second = null. The reversed half is 5-&gt;4.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span><small data-motion-key="pointer-previous">previous</small></span><span class="coding-trace-link-arrow" data-motion-key="link-5-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span></span></div></div><div class="coding-trace-meta"><span><b>firstHalf</b>1-&gt;2-&gt;3-&gt;null</span><span><b>reversed</b>5-&gt;4-&gt;null</span><span><b>secondPointer</b>null</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>reversed</dt><dd><span class="coding-state-before">4-&gt;null</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5-&gt;4-&gt;null</strong></dd></div><div class="coding-state-change"><dt>second pointer</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">null</strong></dd></div><div class="coding-state-change"><dt>saved next</dt><dd><span class="coding-state-before">5</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="initialize-merge" hidden role="group" aria-label="Initialize the merge"><div class="coding-trace-frame-heading"><span>Initialize the merge</span><strong>Set first = head = 1 and second = previous = 5. The two independent chains are 1-&gt;2-&gt;3 and 5-&gt;4.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span><small data-motion-key="pointer-second">second</small></span><span class="coding-trace-link-arrow" data-motion-key="link-5-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span></span></div></div><div class="coding-trace-meta"><span><b>firstHalf</b>1-&gt;2-&gt;3</span><span><b>secondPointer</b>5</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>first half</dt><dd><span class="coding-state-before">1-&gt;2-&gt;3-&gt;null</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1-&gt;2-&gt;3</strong></dd></div><div class="coding-state-change"><dt>second pointer</dt><dd><span class="coding-state-before">null</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5</strong></dd></div><div class="coding-state-change"><dt>reversed</dt><dd><span class="coding-state-before">5-&gt;4-&gt;null</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="7" data-frame-key="merge-node-5" hidden role="group" aria-label="Splice node 5 after node 1"><div class="coding-trace-frame-heading"><span>Splice node 5 after node 1</span><strong>Save first_next = 2 and second_next = 4. Write 1.next = 5 and 5.next = 2, then advance first = 2 and second = 4.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span></span><span class="coding-trace-link-arrow" data-motion-key="link-5-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-2"><span>2</span><small data-motion-key="pointer-first">first</small></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-3"><span>3</span></span></div><div class="coding-trace-linked-row"><span class="coding-trace-label">second</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span></div></div><div class="coding-trace-meta"><span><b>saved</b>first_next=2; second_next=4</span><span><b>secondPointer</b>4</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>saved</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">first_next=2; second_next=4</strong></dd></div><div class="coding-state-change"><dt>second pointer</dt><dd><span class="coding-state-before">5</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div><div class="coding-state-change"><dt>first half</dt><dd><span class="coding-state-before">1-&gt;2-&gt;3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="8" data-frame-key="merge-node-4" hidden role="group" aria-label="Splice node 4 after node 2"><div class="coding-trace-frame-heading"><span>Splice node 4 after node 2</span><strong>Save first_next = 3 and second_next = null. Write 2.next = 4 and 4.next = 3, then second becomes null.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span></span><span class="coding-trace-link-arrow" data-motion-key="link-5-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-3"><span>3</span><small data-motion-key="pointer-first">first</small></span></div></div><div class="coding-trace-meta"><span><b>saved</b>first_next=3; second_next=null</span><span><b>secondPointer</b>null</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>saved</dt><dd><span class="coding-state-before">first_next=2; second_next=4</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">first_next=3; second_next=null</strong></dd></div><div class="coding-state-change"><dt>second pointer</dt><dd><span class="coding-state-before">4</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">null</strong></dd></div><div class="coding-state-change"><dt>node 3</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="9" data-frame-key="finish-reordered-list" hidden role="group" aria-label="Finish when the second half is empty"><div class="coding-trace-frame-heading"><span>Finish when the second half is empty</span><strong>The merge guard fails at second = null. The in-place list is 1-&gt;5-&gt;2-&gt;4-&gt;3-&gt;null.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="list-node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-5"><span>5</span></span><span class="coding-trace-link-arrow" data-motion-key="link-5-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="list-node-3"><span>3</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>1-&gt;5-&gt;2-&gt;4-&gt;3</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1-&gt;5-&gt;2-&gt;4-&gt;3</strong></dd></div><div class="coding-state-change"><dt>saved</dt><dd><span class="coding-state-before">first_next=3; second_next=null</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>second pointer</dt><dd><span class="coding-state-before">null</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 10</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize middle pointers</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Advance middle pointers once</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Advance middle pointers twice</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Detach the second half</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Reverse second-half node 4</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Reverse second-half node 5</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Initialize the merge</strong></button><button type="button" data-coding-frame-button="7"><span>8</span><strong>Splice node 5 after node 1</strong></button><button type="button" data-coding-frame-button="8"><span>9</span><strong>Splice node 4 after node 2</strong></button><button type="button" data-coding-frame-button="9"><span>10</span><strong>Finish when the second half is empty</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when nodes must alternate from the front and back of a singly linked list without an auxiliary array or stack.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">The midpoint scan keeps fast twice as far along; reversal preserves a reversed prefix plus untouched suffix; merging preserves the final alternating prefix and two unconsumed chains.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Complex list transforms become safe compositions of small invariants: locate a boundary, detach, reverse with a saved route, then splice only after saving both continuations.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For 1-&gt;2-&gt;3-&gt;4-&gt;5, set slow = 1 and fast = 1. Both fast.next and fast.next.next exist. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="reorder-list-state"><strong>Check:</strong> Why must 3.next become null before weaving? Where does the odd middle node end up?</span></figcaption></figure>

The slow pointer leaves the first half at least as long as the second.
Detachment prevents an old middle link from forming a cycle.
Save both successors before each splice, so every unprocessed node remains reachable.

## Code

Define the platform node type before the unchanged reference.

```python
class ListNode:
   def __init__(self, val: int = 0, next_node: "ListNode | None" = None):
      self.val = val
      self.next = next_node
```

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

## Change the Requirement

### Preserve the Input

**New task:** Return a reordered copy without changing original links.

**Picture and change:** Copy in original order, then reuse the complete reference on the copy.

**Code:** Use `ListNode` above.

```python
def reordered_copy(head: ListNode | None) -> ListNode | None:
   dummy = tail = ListNode()
   while head:
      tail.next = ListNode(head.val)
      tail = tail.next
      head = head.next
   reorder_list(dummy.next)
   return dummy.next
```

**Check:** `1->2->3` yields fresh `1->3->2` nodes while the input stays `1->2->3`.

**Use when:** External references require the original topology; result allocation becomes $O(n)$.

### Undo the Reordering

**New task:** Recover the original order after the stated transformation.

**Picture and change:** Keep alternating first-half nodes; prepend alternating second-half nodes, then join both chains.

**Code:** No node allocation is required.

```python
def undo_reorder(head: ListNode | None) -> None:
   first = head
   second_reversed = None
   while first and first.next:
      second = first.next
      first.next = second.next
      second.next = second_reversed
      second_reversed = second
      if first.next is None:
         break
      first = first.next
   if first:
      first.next = second_reversed
```

**Check:** `1->5->2->4->3` returns to `1->2->3->4->5`.

**Use when:** The input was produced by the forward transformation.
Time remains linear and workspace constant.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

What happens if `slow.next = None` is omitted?

<details>
<summary>Reasoned answer</summary>

For `1->2->3`, node two retains its old link to node three.
The alternating splice then links node three back to node two, forming a cycle.
Reversal alone does not establish disjoint ownership of the two chains.
Detaching is part of the proof, not optional cleanup.

</details>

</div>
