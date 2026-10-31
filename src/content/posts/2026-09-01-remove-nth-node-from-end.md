---
title: "Keep a link gap until the leading pointer reaches the tail"
description: "Locate the predecessor of an end-relative target, preserve invalid-rank behavior, then switch to read-only lookup or front-index removal."
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

Remove the node at a specified one-based position counted backward from a linked list's end.

## Problem

**Input.** The head of a finite acyclic linked list and an integer rank `n` counted from the end.

**Output.** Return the head of the list after removing that occurrence, which may be a different head.

**Rules.** Rank 1 means the tail; rank equal to the list length means the head.
Rewrite links among retained original nodes rather than copying their values into new nodes.
The reference leaves the chain unchanged for nonpositive or oversized ranks, including an empty chain.

**Example.** Removing rank 2 from `1 -> 2 -> 3 -> 4 -> 5` returns `1 -> 2 -> 3 -> 5`.
Removing rank 1 from a one-node list returns `None`.

## Picture

Start `left` and `right` at a dummy before the head.
Advance `right` by `n` links, then move both references one link per round.
Their gap stays exactly `n` links.

For `n=2`, their positions become `(dummy,2)`, `(1,3)`, `(2,4)`, then `(3,5)`.
Stop when `right.next` is missing: `right` is at the tail, not beyond it.
Now `left.next` is the target, node `4`.

For length $L$, the tail is at position $L$ after the dummy's position zero.
The gap places `left` at $L-n$, immediately before position $L-n+1$.
Assigning `left.next = left.next.next` bypasses that target.
The dummy also supplies a predecessor when the target is the original head.

<!-- visual:remove-nth-node-from-end-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="remove-nth-node-from-end-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="remove-nth-node-from-end-state-title">A fixed pointer gap places left immediately before the node to remove</p><div class="visual-argument" data-visual-argument="remove-nth-node-from-end-state" data-argument-kind="ledger" data-argument-composition="sequence"><svg viewBox="0 0 360 296" role="img" aria-label="Worked example: the dummy predecessor makes deleting the head an ordinary splice. For (1,2,3) and n=3, the predecessor is the dummy node. Its next link skips node 1 and returns head 2."><title>Worked example: the dummy predecessor makes deleting the head an ordinary splice</title><desc>For (1,2,3) and n=3, the predecessor is the dummy node. Its next link skips node 1 and returns head 2.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Before deleting the third from the</tspan><tspan x="20" dy="18">end</tspan></text><g data-proof-token="dummy" data-proof-row="0" data-proof-position="0" data-proof-box="20,78,74,48"><rect x="20" y="78" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="57" y="105" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="57" dy="0">dummy</tspan></text></g><path d="M97 102L101 102" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-proof-token="1" data-proof-row="0" data-proof-position="1" data-proof-box="100,78,74,48"><rect x="100" y="78" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="137" y="105" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="137" dy="0">1</tspan></text></g><path d="M177 102L181 102" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-proof-token="2" data-proof-row="0" data-proof-position="2" data-proof-box="180,78,74,48"><rect x="180" y="78" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="217" y="105" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="217" dy="0">2</tspan></text></g><path d="M257 102L261 102" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-proof-token="3" data-proof-row="0" data-proof-position="3" data-proof-box="260,78,74,48"><rect x="260" y="78" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="297" y="105" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="297" dy="0">3</tspan></text></g><text x="20" y="162" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">After predecessor.next skips the</tspan><tspan x="20" dy="18">target</tspan></text><g data-proof-token="dummy" data-proof-row="1" data-proof-position="0" data-proof-box="20,212,100.66666666666667,48"><rect x="20" y="212" width="100.66666666666667" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="70.33333333333334" y="239" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="70.33333333333334" dy="0">dummy</tspan></text></g><path d="M123.66666666666667 236L127.66666666666667 236" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><g data-proof-token="2" data-proof-row="1" data-proof-position="1" data-proof-box="126.66666666666667,212,100.66666666666667,48"><rect x="126.66666666666667" y="212" width="100.66666666666667" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="177" y="239" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">2</tspan></text></g><path d="M230.33333333333334 236L234.33333333333334 236" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><g data-proof-token="3" data-proof-row="1" data-proof-position="2" data-proof-box="233.33333333333334,212,100.66666666666667,48"><rect x="233.33333333333334" y="212" width="100.66666666666667" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="283.6666666666667" y="239" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="283.6666666666667" dy="0">3</tspan></text></g></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="remove-nth-node-from-end" role="group" tabindex="0" aria-label="Keep a link gap until the leading pointer reaches the tail: Maintain a two-node gap so left stops immediately before the second node from the end."><div class="coding-visual-example"><span>Input and goal</span><strong>Remove the `n`th node counted from the end.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize-dummy" role="group" aria-label="Start both pointers at the dummy"><div class="coding-trace-frame-heading"><span>Start both pointers at the dummy</span><strong>Both pointers start at a dummy before the head. The same bypass can remove the head.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-dummy"><span>dummy</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-dummy-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>input</b>head=[1,2,3,4,5], n=2</span><span><b>right</b>dummy</span><span><b>gap</b>0 nodes</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>input</dt><dd><strong class="coding-state-after">head=[1,2,3,4,5], n=2</strong></dd></div><div class="coding-state-change"><dt>right</dt><dd><strong class="coding-state-after">dummy</strong></dd></div><div class="coding-state-change"><dt>gap</dt><dd><strong class="coding-state-after">0 nodes</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="advance-right-one" hidden role="group" aria-label="Advance right once"><div class="coding-trace-frame-heading"><span>Advance right once</span><strong>The first of n=2 gap steps moves right from dummy to node 1; left stays at dummy.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-dummy"><span>dummy</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-dummy-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span><small data-motion-key="pointer-right">right</small></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>input</b>head=[1,2,3,4,5], n=2</span><span><b>gap</b>1 node</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>gap</dt><dd><span class="coding-state-before">0 nodes</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1 node</strong></dd></div><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before">dummy</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="advance-right-two" hidden role="group" aria-label="Finish the two-node gap"><div class="coding-trace-frame-heading"><span>Finish the two-node gap</span><strong>The second gap step moves right to node 2. Right is now exactly two links ahead of left.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-dummy"><span>dummy</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-dummy-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span><small data-motion-key="pointer-right">right</small></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>input</b>head=[1,2,3,4,5], n=2</span><span><b>gap</b>2 nodes</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>gap</dt><dd><span class="coding-state-before">1 node</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2 nodes</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="walk-together-one" hidden role="group" aria-label="Move both pointers: step 1"><div class="coding-trace-frame-heading"><span>Move both pointers: step 1</span><strong>Because right.next is node 3, move left to 1 and right to 3 while preserving the two-link gap.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-dummy"><span>dummy</span></span><span class="coding-trace-link-arrow" data-motion-key="link-dummy-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span><small data-motion-key="pointer-right">right</small></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>input</b>head=[1,2,3,4,5], n=2</span><span><b>gap</b>2 nodes</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="walk-together-two" hidden role="group" aria-label="Move both pointers: step 2"><div class="coding-trace-frame-heading"><span>Move both pointers: step 2</span><strong>Because right.next is node 4, move left to 2 and right to 4; the gap remains two links.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-dummy"><span>dummy</span></span><span class="coding-trace-link-arrow" data-motion-key="link-dummy-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span><small data-motion-key="pointer-right">right</small></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>input</b>head=[1,2,3,4,5], n=2</span><span><b>gap</b>2 nodes</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="walk-together-three" hidden role="group" aria-label="Move both pointers: step 3"><div class="coding-trace-frame-heading"><span>Move both pointers: step 3</span><strong>Because right.next is node 5, move left to 3 and right to 5. Now right.next is null, so the loop stops.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-dummy"><span>dummy</span></span><span class="coding-trace-link-arrow" data-motion-key="link-dummy-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-3"><span>3</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-3-4">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-4"><span>4</span></span><span class="coding-trace-link-arrow" data-motion-key="link-4-5">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-5"><span>5</span><small data-motion-key="pointer-right">right</small></span></div></div><div class="coding-trace-meta"><span><b>input</b>head=[1,2,3,4,5], n=2</span><span><b>gap</b>2 nodes</span><span><b>target</b>left.next is node 4</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>target</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">left.next is node 4</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="remove-four" hidden role="group" aria-label="Rewire around node 4"><div class="coding-trace-frame-heading"><span>Rewire around node 4</span><strong>Set left.next = left.next.next, changing the link 3-&gt;4 into 3-&gt;5. Return dummy.next at node 1.</strong></div><div class="coding-trace-linked"><div class="coding-trace-linked-row"><span class="coding-trace-linked-node" data-motion-key="node-dummy"><span>dummy</span></span><span class="coding-trace-link-arrow" data-motion-key="link-dummy-1">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-1"><span>1</span></span><span class="coding-trace-link-arrow" data-motion-key="link-1-2">&rarr;</span><span class="coding-trace-linked-node" data-motion-key="node-2"><span>2</span></span><span class="coding-trace-link-arrow" data-motion-key="link-2-3">&rarr;</span><span class="coding-trace-linked-node trace-tone-focus" data-motion-key="node-3"><span>3</span><small data-motion-key="pointer-left">left</small></span><span class="coding-trace-link-arrow" data-motion-key="link-3-5">&rarr;</span><span class="coding-trace-linked-node trace-tone-output" data-motion-key="node-5"><span>5</span></span></div></div><div class="coding-trace-meta"><span><b>changedLink</b>3.next: node 4 -&gt; node 5</span><span><b>removed</b>node 4</span><span><b>result</b>[1,2,3,5]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>changed link</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3.next: node 4 -&gt; node 5</strong></dd></div><div class="coding-state-change"><dt>removed</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">node 4</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[1,2,3,5]</strong></dd></div></dl><div class="coding-numeric-transition"><svg viewBox="0 0 360 158" role="img" aria-label="node 4: 4 before, 5 at the next shown checkpoint." style="display:block;width:100%;max-width:34rem;margin:auto"><title>node 4: 4 before, 5 at the next shown checkpoint.</title><desc>Each row has its own zero-inclusive numerical scale. Circles mark the preceding authored checkpoint and squares the current checkpoint. Connecting segments compare these shown values, not an invented path of execution.</desc><text x="16" y="20" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text-soft)">SHOWN CHECKPOINTS</text><text x="344" y="20" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text-soft)">circle: before / square: now</text><g data-numeric-transition="node:4" data-before="4" data-after="5" data-minimum="0" data-maximum="5"><text x="16" y="45" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">node 4</text><text x="344" y="45" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">4 -&gt; 5</text><path d="M28 96H332" style="stroke:var(--c-rule);fill:none"></path><path d="M28 65V111" style="stroke:var(--c-text-soft);stroke-dasharray:3 3;fill:none"></path><path d="M271.20000000000005 69L332 96" style="stroke:var(--viz-focus-stroke);stroke-width:2;fill:none"></path><circle data-checkpoint-before cx="271.20000000000005" cy="69" r="5" style="fill:var(--c-bg);stroke:var(--c-text-soft);stroke-width:2"></circle><rect data-checkpoint-after x="327" y="91" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="28" y="128" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">0</text><text x="332" y="128" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">5</text></g></svg></div></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 7</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start both pointers at the dummy</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Advance right once</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Finish the two-node gap</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Move both pointers: step 1</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Move both pointers: step 2</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Move both pointers: step 3</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Rewire around node 4</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">The node is identified relative to the list end, but only one forward pass is allowed, so a leading pointer can convert distance-from-end into a simultaneous stopping position.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After the initial advance, right remains exactly n links ahead of left. Therefore, when right is the tail, left.next is the nth node from the end.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Use a fixed lead whenever a linked-list target is defined by distance from the end; add a dummy when the operation may change the head.</dd></div></dl></div><div class="visual-derivation-caption">Both pointers start at a dummy before the head. The same bypass can remove the head. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="remove-nth-node-from-end-state"><strong>Check:</strong> Why is left at node 3 when right reaches node 5? Which single link removes node 4?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="remove-nth-node-from-end-state">For (1,2,3) and n=3, the predecessor is the dummy node. Its next link skips node 1 and returns head 2.</span></figcaption></figure>

## Code

The platform supplies `ListNode`; this definition supports all blocks when running them together locally.

```python
class ListNode:
   def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
```

```python
def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
   dummy = ListNode(0, head)
   left = dummy
   right = dummy

   for _ in range(n):
      if right.next is None:
         return head
      right = right.next

   while right.next:
      left = left.next
      right = right.next

   if left.next:
      left.next = left.next.next
   return dummy.next
```

Let $L$ denote list length, distinct from rank `n`.
The function performs $O(L)$ link work using $O(1)$ extra node references.
Empty input returns `None`.

### Failure Check

Changing the initial gap to `n+1` without changing the stopping rule selects the wrong predecessor.
Bypassing a node removes it from the returned chain; it does not destroy external references or clear its outgoing link.

<!-- learning:core-end -->

## Change to Read-Only Lookup

**New task:** Return the nth value from the end without changing links.

**Picture and change:** Lead by `rank` links from the head, then walk until the lead reaches `None`.
No predecessor is needed, so both the starting and stopping conventions change.

```python
def nth_value_from_end(head, rank):
   if rank <= 0:
      return None
   lead = head
   for _ in range(rank):
      if lead is None:
         return None
      lead = lead.next
   lag = head
   while lead:
      lead, lag = lead.next, lag.next
   return lag.val
```

**Check:** The example returns `4` with all links unchanged; invalid ranks return `None`.
If `None` is a valid payload, choose a different missing-result contract.

**Use when:** The end-relative query is read-only.

## Change to a Front Index

**New task:** Remove a node identified by its zero-based front index.

**Picture and change:** Count length, convert `index` to end rank `length-index`, then call the reference above.

```python
def remove_front_index(head, index):
   length, node = 0, head
   while node:
      length += 1
      node = node.next
   if not 0 <= index < length:
      return head
   return remove_nth_from_end(head, length - index)
```

**Check:** Index `0` on `[1,2,3]` maps to rank `3` and returns the original `2 -> 3` suffix.
Index `4` leaves that input unchanged.
This adapter makes two passes; both variants retain linear link work and constant extra references.

**Use when:** The interface supplies front-relative coordinates.

### Transfer Check

Why does removal start at a dummy while value lookup starts at the head?

### Answer

Removal needs an incoming link to change, even for the head.
Lookup needs the target itself, so it uses a different boundary convention.
