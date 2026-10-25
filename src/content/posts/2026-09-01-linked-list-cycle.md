---
title: "Detect a cycle with one-link and two-link moves"
description: "Follow actual node identities, prove the relative-offset collision, then recover the cycle entry and length without changing links."
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

Two references moving at different speeds eventually identify the same node inside a cycle; equal stored values prove nothing.

## Problem

Check whether following `next` from a head ever revisits a node.
For `1 -> 2 -> 3 -> 4` with `4.next` pointing to node `2`, return `True`.
Do not change links or store every visited node.
Use ordinary `ListNode` objects with stable `next` links; an empty list returns `False`.

## Picture

Let `slow` follow one link per round and `fast` follow two.
Both start at the head, so initial equality cannot distinguish a cycle from an ordinary list.
Compare only after moving.

| Completed round | `slow` node | `fast` node |
| --- | --- | --- |
| 0 | 1 | 1; not compared |
| 1 | 2 | 3 |
| 2 | 3 | 2 |
| 3 | 4 | 4; identity collision |

Once both references enter a cycle of length $c$, their relative offset changes by one modulo $c$ each round.
It reaches zero within $c$ further rounds.
Without a cycle, `fast` eventually reaches a missing link instead.

<!-- visual:linked-list-cycle-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="linked-list-cycle-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="linked-list-cycle-state-title">Detect a cycle with one-link and two-link moves: With a cycle, a pointer moving two links per round gains one cycle position on a pointer moving one link, so they must meet.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="linked-list-cycle" role="group" tabindex="0" aria-label="Detect a cycle with one-link and two-link moves: With a cycle, a pointer moving two links per round gains one cycle position on a pointer moving one link, so they must meet."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether a linked list contains a cycle.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize-at-head" role="group" aria-label="Initialize both pointers at head"><div class="coding-trace-frame-heading"><span>Initialize both pointers at head</span><strong>Start both pointers at node 1 without comparing them. Guard the two-link move before advancing.</strong></div><div class="coding-trace-linked-topology"><svg viewBox="0 0 420 190" role="img" aria-label="Linked nodes and next-pointer edges"><title>Linked-list topology</title><desc>Nodes, next pointers, and moving algorithm pointers.</desc><defs><marker id="coding-linked-arrow-0" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker><marker id="coding-linked-arrow-0-warning" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker is-warning" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-topology-row-label" x="8" y="78">next</text><g class="coding-trace-topology-edge" data-motion-key="edge-1-2"><path d="M 90 74 L 127 74" marker-end="url(#coding-linked-arrow-0)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-2-3"><path d="M 186 74 L 223 74" marker-end="url(#coding-linked-arrow-0)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-3-4"><path d="M 282 74 L 319 74" marker-end="url(#coding-linked-arrow-0)" /></g><g class="coding-trace-topology-edge trace-tone-warning" data-motion-key="edge-4-2"><path d="M 350 92 C 350 156 158 156 158 96" marker-end="url(#coding-linked-arrow-0-warning)" /><text class="coding-trace-topology-edge-label" x="254" y="168">4.next -&gt; 2</text></g><g class="coding-trace-topology-node trace-tone-output" data-motion-key="node-1"><text class="coding-trace-topology-pointer" x="62" y="49" data-motion-key="pointer-slow">slow</text><text class="coding-trace-topology-pointer" x="62" y="36" data-motion-key="pointer-fast">fast</text><rect x="37" y="57" width="50" height="34" rx="3" /><text x="62" y="79">1</text></g><g class="coding-trace-topology-node" data-motion-key="node-2"><rect x="133" y="57" width="50" height="34" rx="3" /><text x="158" y="79">2</text></g><g class="coding-trace-topology-node" data-motion-key="node-3"><rect x="229" y="57" width="50" height="34" rx="3" /><text x="254" y="79">3</text></g><g class="coding-trace-topology-node" data-motion-key="node-4"><rect x="325" y="57" width="50" height="34" rx="3" /><text x="350" y="79">4</text></g></svg></div><div class="coding-trace-meta"><span><b>slowAt</b>1</span><span><b>fastAt</b>1</span><span><b>guard</b>fast=1 and fast.next=2</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="advance-round-1" hidden role="group" aria-label="Advance round 1"><div class="coding-trace-frame-heading"><span>Advance round 1</span><strong>Move slow one link 1-&gt;2. Move fast two links 1-&gt;2-&gt;3. They are different nodes, so continue.</strong></div><div class="coding-trace-linked-topology"><svg viewBox="0 0 420 190" role="img" aria-label="Linked nodes and next-pointer edges"><title>Linked-list topology</title><desc>Nodes, next pointers, and moving algorithm pointers.</desc><defs><marker id="coding-linked-arrow-1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker><marker id="coding-linked-arrow-1-warning" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker is-warning" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-topology-row-label" x="8" y="78">next</text><g class="coding-trace-topology-edge" data-motion-key="edge-1-2"><path d="M 90 74 L 127 74" marker-end="url(#coding-linked-arrow-1)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-2-3"><path d="M 186 74 L 223 74" marker-end="url(#coding-linked-arrow-1)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-3-4"><path d="M 282 74 L 319 74" marker-end="url(#coding-linked-arrow-1)" /></g><g class="coding-trace-topology-edge trace-tone-warning" data-motion-key="edge-4-2"><path d="M 350 92 C 350 156 158 156 158 96" marker-end="url(#coding-linked-arrow-1-warning)" /><text class="coding-trace-topology-edge-label" x="254" y="168">4.next -&gt; 2</text></g><g class="coding-trace-topology-node" data-motion-key="node-1"><rect x="37" y="57" width="50" height="34" rx="3" /><text x="62" y="79">1</text></g><g class="coding-trace-topology-node trace-tone-focus" data-motion-key="node-2"><text class="coding-trace-topology-pointer" x="158" y="49" data-motion-key="pointer-slow">slow</text><rect x="133" y="57" width="50" height="34" rx="3" /><text x="158" y="79">2</text></g><g class="coding-trace-topology-node trace-tone-state" data-motion-key="node-3"><text class="coding-trace-topology-pointer" x="254" y="49" data-motion-key="pointer-fast">fast</text><rect x="229" y="57" width="50" height="34" rx="3" /><text x="254" y="79">3</text></g><g class="coding-trace-topology-node" data-motion-key="node-4"><rect x="325" y="57" width="50" height="34" rx="3" /><text x="350" y="79">4</text></g></svg></div><div class="coding-trace-meta"><span><b>slowAt</b>2</span><span><b>fastAt</b>3</span><span><b>movement</b>slow: 1-&gt;2; fast: 1-&gt;2-&gt;3</span><span><b>identityCheck</b>2 is not 3</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="advance-round-2" hidden role="group" aria-label="Advance round 2"><div class="coding-trace-frame-heading"><span>Advance round 2</span><strong>Move slow 2-&gt;3. Move fast 3-&gt;4-&gt;2, wrapping through the cycle. They still differ.</strong></div><div class="coding-trace-linked-topology"><svg viewBox="0 0 420 190" role="img" aria-label="Linked nodes and next-pointer edges"><title>Linked-list topology</title><desc>Nodes, next pointers, and moving algorithm pointers.</desc><defs><marker id="coding-linked-arrow-2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker><marker id="coding-linked-arrow-2-warning" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker is-warning" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-topology-row-label" x="8" y="78">next</text><g class="coding-trace-topology-edge" data-motion-key="edge-1-2"><path d="M 90 74 L 127 74" marker-end="url(#coding-linked-arrow-2)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-2-3"><path d="M 186 74 L 223 74" marker-end="url(#coding-linked-arrow-2)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-3-4"><path d="M 282 74 L 319 74" marker-end="url(#coding-linked-arrow-2)" /></g><g class="coding-trace-topology-edge trace-tone-warning" data-motion-key="edge-4-2"><path d="M 350 92 C 350 156 158 156 158 96" marker-end="url(#coding-linked-arrow-2-warning)" /><text class="coding-trace-topology-edge-label" x="254" y="168">4.next -&gt; 2</text></g><g class="coding-trace-topology-node" data-motion-key="node-1"><rect x="37" y="57" width="50" height="34" rx="3" /><text x="62" y="79">1</text></g><g class="coding-trace-topology-node trace-tone-state" data-motion-key="node-2"><text class="coding-trace-topology-pointer" x="158" y="49" data-motion-key="pointer-fast">fast</text><rect x="133" y="57" width="50" height="34" rx="3" /><text x="158" y="79">2</text></g><g class="coding-trace-topology-node trace-tone-focus" data-motion-key="node-3"><text class="coding-trace-topology-pointer" x="254" y="49" data-motion-key="pointer-slow">slow</text><rect x="229" y="57" width="50" height="34" rx="3" /><text x="254" y="79">3</text></g><g class="coding-trace-topology-node" data-motion-key="node-4"><rect x="325" y="57" width="50" height="34" rx="3" /><text x="350" y="79">4</text></g></svg></div><div class="coding-trace-meta"><span><b>slowAt</b>3</span><span><b>fastAt</b>2</span><span><b>movement</b>slow: 2-&gt;3; fast: 3-&gt;4-&gt;2</span><span><b>identityCheck</b>3 is not 2</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="advance-round-3" hidden role="group" aria-label="Advance round 3"><div class="coding-trace-frame-heading"><span>Advance round 3</span><strong>Move slow 3-&gt;4. Move fast 2-&gt;3-&gt;4. Both object references now identify node 4.</strong></div><div class="coding-trace-linked-topology"><svg viewBox="0 0 420 190" role="img" aria-label="Linked nodes and next-pointer edges"><title>Linked-list topology</title><desc>Nodes, next pointers, and moving algorithm pointers.</desc><defs><marker id="coding-linked-arrow-3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker><marker id="coding-linked-arrow-3-warning" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker is-warning" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-topology-row-label" x="8" y="78">next</text><g class="coding-trace-topology-edge" data-motion-key="edge-1-2"><path d="M 90 74 L 127 74" marker-end="url(#coding-linked-arrow-3)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-2-3"><path d="M 186 74 L 223 74" marker-end="url(#coding-linked-arrow-3)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-3-4"><path d="M 282 74 L 319 74" marker-end="url(#coding-linked-arrow-3)" /></g><g class="coding-trace-topology-edge trace-tone-warning" data-motion-key="edge-4-2"><path d="M 350 92 C 350 156 158 156 158 96" marker-end="url(#coding-linked-arrow-3-warning)" /><text class="coding-trace-topology-edge-label" x="254" y="168">4.next -&gt; 2</text></g><g class="coding-trace-topology-node" data-motion-key="node-1"><rect x="37" y="57" width="50" height="34" rx="3" /><text x="62" y="79">1</text></g><g class="coding-trace-topology-node" data-motion-key="node-2"><rect x="133" y="57" width="50" height="34" rx="3" /><text x="158" y="79">2</text></g><g class="coding-trace-topology-node" data-motion-key="node-3"><rect x="229" y="57" width="50" height="34" rx="3" /><text x="254" y="79">3</text></g><g class="coding-trace-topology-node trace-tone-output" data-motion-key="node-4"><text class="coding-trace-topology-pointer" x="350" y="49" data-motion-key="pointer-slow">slow</text><text class="coding-trace-topology-pointer" x="350" y="36" data-motion-key="pointer-fast">fast</text><rect x="325" y="57" width="50" height="34" rx="3" /><text x="350" y="79">4</text></g></svg></div><div class="coding-trace-meta"><span><b>slowAt</b>4</span><span><b>fastAt</b>4</span><span><b>movement</b>slow: 3-&gt;4; fast: 2-&gt;3-&gt;4</span><span><b>identityCheck</b>slow is fast at node 4</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="return-true" hidden role="group" aria-label="Return true on identity"><div class="coding-trace-frame-heading"><span>Return true on identity</span><strong>The in-loop identity check succeeds, so return true before another guard evaluation.</strong></div><div class="coding-trace-linked-topology"><svg viewBox="0 0 420 190" role="img" aria-label="Linked nodes and next-pointer edges"><title>Linked-list topology</title><desc>Nodes, next pointers, and moving algorithm pointers.</desc><defs><marker id="coding-linked-arrow-4" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker><marker id="coding-linked-arrow-4-warning" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker is-warning" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><text class="coding-trace-topology-row-label" x="8" y="78">next</text><g class="coding-trace-topology-edge" data-motion-key="edge-1-2"><path d="M 90 74 L 127 74" marker-end="url(#coding-linked-arrow-4)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-2-3"><path d="M 186 74 L 223 74" marker-end="url(#coding-linked-arrow-4)" /></g><g class="coding-trace-topology-edge" data-motion-key="edge-3-4"><path d="M 282 74 L 319 74" marker-end="url(#coding-linked-arrow-4)" /></g><g class="coding-trace-topology-edge trace-tone-warning" data-motion-key="edge-4-2"><path d="M 350 92 C 350 156 158 156 158 96" marker-end="url(#coding-linked-arrow-4-warning)" /><text class="coding-trace-topology-edge-label" x="254" y="168">4.next -&gt; 2</text></g><g class="coding-trace-topology-node" data-motion-key="node-1"><rect x="37" y="57" width="50" height="34" rx="3" /><text x="62" y="79">1</text></g><g class="coding-trace-topology-node" data-motion-key="node-2"><rect x="133" y="57" width="50" height="34" rx="3" /><text x="158" y="79">2</text></g><g class="coding-trace-topology-node" data-motion-key="node-3"><rect x="229" y="57" width="50" height="34" rx="3" /><text x="254" y="79">3</text></g><g class="coding-trace-topology-node trace-tone-output" data-motion-key="node-4"><text class="coding-trace-topology-pointer" x="350" y="49" data-motion-key="pointer-slow">slow</text><text class="coding-trace-topology-pointer" x="350" y="36" data-motion-key="pointer-fast">fast</text><rect x="325" y="57" width="50" height="34" rx="3" /><text x="350" y="79">4</text></g></svg></div><div class="coding-trace-meta"><span><b>slowAt</b>4</span><span><b>fastAt</b>4</span><span><b>check</b>slow is fast</span><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize both pointers at head</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Advance round 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Advance round 2</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Advance round 3</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Return true on identity</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it for cycle detection in a deterministic next-pointer structure when constant extra space is required and storing every visited node would be unnecessary.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After r loop iterations, slow has followed r next links and fast has followed 2r. If no cycle exists fast reaches null; inside a cycle their relative offset changes by one per round.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Different traversal speeds convert a hidden cycle into an inevitable identity collision; guard every fast hop. The technique transfers to repeated-state sequences and cycle-entry algorithms.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Start both pointers at node 1 without comparing them. Guard the two-link move before advancing. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

The platform supplies this node contract; use the definition below when running these blocks together locally.

```python
class ListNode:
   def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
```

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

The guard makes both link dereferences safe; the second hop may return `None`.

`is` compares object identity, not stored values or custom equality.
For $n$ reachable nodes, the function uses $O(n)$ link operations and $O(1)$ extra storage.

### Failure Check

Two distinct acyclic nodes both valued `7` still return `False`.
Comparing values, or comparing the shared head before movement, would give a false cycle certificate.

<!-- learning:core-end -->

## Change to Entry Identity

**New task:** Return the original node where the cycle begins, or `None`.

**Picture and change:** After collision, reset one reference to the head and advance both by one link.
At collision, the faster reference's extra distance is a multiple of the cycle length.
If the entry is $\mu$ links from the head, another $\mu$ links from the collision also reaches that entry.

```python
def cycle_entry(head):
   slow = fast = head
   while fast and fast.next:
      slow, fast = slow.next, fast.next.next
      if slow is fast:
         entry = head
         while entry is not slow:
            entry, slow = entry.next, slow.next
         return entry
   return None
```

**Check:** The example returns the original node `2`, not a new node with value `2`.
The collision occurs at node `4`, showing that collision and entry need not coincide.

**Use when:** The repeated component needs a node-identity witness.

## Change to Cycle Length

**New task:** Count distinct nodes in the cycle, excluding the noncyclic prefix.

**Picture and change:** Use `cycle_entry` above, then count one traversal back to that same node.

```python
def cycle_length(head):
   entry = cycle_entry(head)
   if entry is None:
      return 0
   length, node = 1, entry.next
   while node is not entry:
      length += 1
      node = node.next
   return length
```

**Check:** The example returns `3`; a self-loop returns `1`, and an acyclic list returns `0`.
Both adaptations retain linear link work, constant extra storage and all original links.

**Use when:** The repeated-state period is the required quantity.

### Transfer Check

Why is gaining exactly one relative link useful inside the cycle?

### Answer

Successive offsets visit every residue modulo its length, including zero.
The proof concerns positions on next links, not numeric values or visual proximity.
