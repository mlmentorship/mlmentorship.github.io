---
title: "Split Array Largest Sum"
description: "Split an array into `k` nonempty continuous parts. Make the largest part sum as small as possible."
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

Split an array into a fixed number of contiguous groups while making the largest group sum as small as possible.

## Problem

**Input.** A nonempty list `nums` of nonnegative integers and an integer `parts` with `1 <= parts <= len(nums)`.

**Output.** Return the smallest achievable value of the largest group sum across all valid partitions.

**Rules.** Use exactly `parts` groups, each nonempty and contiguous, preserving the original order.
Every input occurrence belongs to exactly one group. Zeros and duplicate values are allowed.
Do not modify the list. Return only the optimal largest sum, not the split positions.

**Example.** `[7, 2, 5, 10, 8]` with `parts = 2` returns `18`.
The split `[7, 2, 5]` and `[10, 8]` has sums 14 and 18; no split lowers the largest sum further.

## Picture

For `[7,2,5,10,8]` and two groups, capacity `17` needs three groups: `[7,2,5]`, `[10]`, `[8]`.
Capacity `18` allows `[7,2,5]`, `[10,8]`.

<!-- visual:split-array-largest-sum-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="split-array-largest-sum-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="split-array-largest-sum-state-title">A failed capacity rules out every smaller capacity for these nonnegative values</p><div class="visual-argument" data-visual-argument="split-array-largest-sum-state" data-argument-kind="bars" data-argument-composition="endpoint"><svg viewBox="0 0 360 327" role="img" aria-label="Worked example: the capacity predicate becomes feasible at the correct boundary. For (7,2,5,10,8), capacity 17 needs three groups; capacity 18 needs two. With k=2, choose the first feasible capacity, 18."><title>Worked example: the capacity predicate becomes feasible at the correct boundary</title><desc>For (7,2,5,10,8), capacity 17 needs three groups; capacity 18 needs two. With k=2, choose the first feasible capacity, 18.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Capacity 17: greedy group count</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="3" data-proof-min="0" data-proof-max="3" data-proof-zero="28" x="28" y="46" width="304" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><circle cx="332" cy="63.5" r="4" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path data-proof-overflow="1" d="M230.66666666666666 54H332M230.66666666666666 63.5H332M230.66666666666666 73H332" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2"></path><path d="M230.66666666666666 40L230.66666666666666 89" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M332 69L332 86" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="296" y="106" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="296" dy="0">3 groups</tspan></text><text x="28" y="131" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="28" dy="0">allowed 2 groups</tspan></text><text x="18" y="165" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Capacity 18: greedy group count</tspan></text><path d="M28 206L332 206" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="2" data-proof-min="0" data-proof-max="3" data-proof-zero="28" x="28" y="188" width="202.66666666666666" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="230.66666666666666" cy="205.5" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M230.66666666666666 182L230.66666666666666 231" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M230.66666666666666 211L230.66666666666666 228" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="230.66666666666666" y="248" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="230.66666666666666" dy="0">2 groups</tspan></text><text x="28" y="273" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="28" dy="0">allowed 2 groups</tspan></text><text x="28" y="307" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="307" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">3</tspan></text></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="split-array-largest-sum" role="group" tabindex="0" aria-label="Split Array Largest Sum: Binary-search the first feasible capacity because the greedy number of contiguous parts never increases when capacity grows."><div class="coding-visual-example"><span>Input and goal</span><strong>Split an array into `k` nonempty continuous parts. Make the largest part sum as small as possible.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize answer bounds"><div class="coding-trace-frame-heading"><span>Initialize answer bounds</span><strong>Capacity must cover the largest value, 10. Capacity 32 fits the whole array.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">10</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>bounds</b>lo = 10, hi = 32</span><span><b>midpoint</b>(10 + 32) // 2 = 21</span><span><b>partition</b>not tested yet</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>bounds</dt><dd><strong class="coding-state-after">lo = 10, hi = 32</strong></dd></div><div class="coding-state-change"><dt>midpoint</dt><dd><strong class="coding-state-after">(10 + 32) // 2 = 21</strong></dd></div><div class="coding-state-change"><dt>partition</dt><dd><strong class="coding-state-after">not tested yet</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="test-21" hidden role="group" aria-label="Test limit 21"><div class="coding-trace-frame-heading"><span>Test limit 21</span><strong>Greedy accumulates 7+2+5=14, then 14+10 exceeds 21, so it cuts before 10; [7,2,5] | [10,8] uses 2 parts and is feasible.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="greedy-cut">cut before</span><span class="coding-trace-array-cell">10</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>partition</b>[7,2,5] | [10,8]</span><span><b>sums</b>14, 18</span><span><b>decision</b>2 &lt;= k, so hi: 32 -&gt; 21</span><span><b>nextBounds</b>lo = 10, hi = 21</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>partition</dt><dd><span class="coding-state-before">not tested yet</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[7,2,5] | [10,8]</strong></dd></div><div class="coding-state-change"><dt>sums</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">14, 18</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="test-15" hidden role="group" aria-label="Test limit 15"><div class="coding-trace-frame-heading"><span>Test limit 15</span><strong>Greedy cuts before 10 because 14+10 exceeds 15, then before 8 because 10+8 exceeds 15; three parts are too many.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="greedy-cut-1">cut before</span><span class="coding-trace-array-cell">10</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-warning" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="greedy-cut-2">cut before</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>midpoint</b>(10 + 21) // 2 = 15</span><span><b>partition</b>[7,2,5] | [10] | [8]</span><span><b>sums</b>14, 10, 8</span><span><b>decision</b>3 &gt; k, so lo: 10 -&gt; 16</span><span><b>nextBounds</b>lo = 16, hi = 21</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="test-18" hidden role="group" aria-label="Test limit 18"><div class="coding-trace-frame-heading"><span>Test limit 18</span><strong>The first cut occurs before 10 and [10,8] reaches exactly 18; two parts are feasible, so keep 18 and discard larger limits.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="greedy-cut">cut before</span><span class="coding-trace-array-cell">10</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>midpoint</b>(16 + 21) // 2 = 18</span><span><b>partition</b>[7,2,5] | [10,8]</span><span><b>sums</b>14, 18</span><span><b>decision</b>2 &lt;= k, so hi: 21 -&gt; 18</span><span><b>nextBounds</b>lo = 16, hi = 18</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before">4</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="test-17" hidden role="group" aria-label="Reject limit 17"><div class="coding-trace-frame-heading"><span>Reject limit 17</span><strong>After the cut before 10, adding 8 would make 18 &gt; 17, forcing a third part; therefore 17 and every smaller candidate are infeasible.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="greedy-cut-1">cut before</span><span class="coding-trace-array-cell">10</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-warning" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="greedy-cut-2">cut before</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>midpoint</b>(16 + 18) // 2 = 17</span><span><b>partition</b>[7,2,5] | [10] | [8]</span><span><b>sums</b>14, 10, 8</span><span><b>decision</b>3 &gt; k, so lo: 16 -&gt; 18</span><span><b>nextBounds</b>lo = 18, hi = 18</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="result" hidden role="group" aria-label="Return the first feasible limit"><div class="coding-trace-frame-heading"><span>Return the first feasible limit</span><strong>The search converges at lo = hi = 18. Greedy proves 18 feasible, while the rejected 17 proves no smaller maximum part sum works.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="greedy-cut">optimal cut</span><span class="coding-trace-array-cell">10</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>partition</b>[7,2,5] | [10,8]</span><span><b>bounds</b>lo = hi = 18</span><span><b>result</b>18</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>optimal cut</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>cut before</dt><dd><span class="coding-state-before">4</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize answer bounds</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Test limit 21</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Test limit 15</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Test limit 18</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Reject limit 17</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Return the first feasible limit</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when asked to minimize a numeric capacity or maximum load, and a candidate limit can be checked greedily with feasibility changing only once as the limit increases.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Every limit below lo is infeasible, at least one feasible answer lies at or below hi, and parts_needed(limit) is nonincreasing; each midpoint decision preserves the smallest feasible limit inside [lo, hi].</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For minimize-the-maximum problems, search the answer when a greedy capacity check is monotone: feasible moves the upper bound down, infeasible moves the lower bound above the candidate.</dd></div></dl></div><div class="visual-derivation-caption">Capacity must cover the largest value, 10. Capacity 32 fits the whole array. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="split-array-largest-sum-state"><strong>Check:</strong> Why does needing three parts at capacity seventeen rule out every smaller capacity under the stated assumptions?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="split-array-largest-sum-state">For (7,2,5,10,8), capacity 17 needs three groups; capacity 18 needs two. With k=2, choose the first feasible capacity, 18.</span></figcaption></figure>

With nonnegative entries, extending each group as far as capacity permits minimizes the required group count.
Increasing capacity cannot increase that count.
Any valid grouping with fewer groups can be split further without increasing its largest sum.
Therefore, testing “at most” groups also solves the exact-count task.

## Code

```python
def split_array_largest_sum(nums: list[int], parts: int) -> int:
   def parts_needed(limit: int) -> int:
      used = 1
      total = 0
      for num in nums:
         if total + num > limit:
            used += 1
            total = 0
         total += num
      return used

   left, right = max(nums), sum(nums)
   while left < right:
      middle = (left + right) // 2
      if parts_needed(middle) <= parts:
         right = middle
      else:
         left = middle + 1
   return left
```

**Cost:** $O(n\log(S-M+2))$ scalar operations and constant workspace, where $S=\sum nums$ and $M=\max nums$.
Integer bit costs are excluded.

## Change the Requirement

### Return the Exact Groups

**New task:** Return a witness with exactly `parts` groups.

**Picture and change:** Find capacity, form maximal groups, then split groups until the required count is reached.

**Code:** Retain the original nonnegative-input contract.

```python
def split_with_groups(nums: list[int], parts: int):
   limit = split_array_largest_sum(nums, parts)
   groups, current, total = [], [], 0
   for value in nums:
      if current and total + value > limit:
         groups.append(current)
         current, total = [], 0
      current.append(value)
      total += value
   groups.append(current)
   extra = parts - len(groups)
   result = []
   for group in groups:
      take = min(extra, len(group) - 1)
      result.extend([[value] for value in group[:take]])
      result.append(group[take:])
      extra -= take
   return limit, result
```

**Check:** `split_with_groups([0,0,0], 3)` returns `(0, [[0], [0], [0]])`.

**Use when:** Boundaries are required; witness storage adds $O(n)$.

### Permit Negative Entries

**New task:** Minimize the same maximum with signed integers.

**Picture and change:** Replace greedy capacity tests with exact prefix-partition dynamic programming.

**Code:** Compare every possible last-group boundary.

```python
def split_signed(nums: list[int], parts: int) -> int:
   if not 1 <= parts <= len(nums):
      raise ValueError("parts must be between one and the input length")
   prefix = [0]
   for value in nums:
      prefix.append(prefix[-1] + value)
   previous = [float("inf")] * (len(nums) + 1)
   previous[0] = -float("inf")
   for count in range(1, parts + 1):
      current = [float("inf")] * (len(nums) + 1)
      for end in range(count, len(nums) + 1):
         current[end] = min(
            max(previous[cut], prefix[end] - prefix[cut])
            for cut in range(count - 1, end)
         )
      previous = current
   return previous[-1]
```

**Check:** `split_signed([5,-4], 1)` returns `1`; the original lower bound `max(nums)=5` is invalid.

**Use when:** Negative entries invalidate greedy reasoning.
Costs become $O(parts\times n^2)$ time and $O(n)$ workspace.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why initialize the empty-prefix maximum to negative infinity in the signed version?

<details>
<summary>Reasoned answer</summary>

Before any group exists, its maximum must not constrain a real negative group sum.
Starting at zero would incorrectly change `split_signed([-5,-4], 1)` from `-9` to `0`.
Positive infinity marks unreachable states; finite reachable results remain integers.

</details>

</div>
