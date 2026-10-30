---
title: "Spiral Matrix"
description: "Return matrix values in spiral order."
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

List every matrix cell in clockwise spiral order, beginning at the top-left corner.

## Problem

**Input.** A rectangular integer `matrix`, possibly with an empty dimension.

**Output.** Return a flat list of all cell values in clockwise spiral order.

**Rules.** Begin at the top-left and initially move right, following successive outer boundaries inward.
Visit every cell exactly once, including repeated or negative values.
Leave the matrix unchanged. Empty dimensions return `[]`; ragged rows are outside the contract.

**Example.** `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` returns `[1, 2, 3, 6, 9, 8, 7, 4, 5]`.
A one-row matrix `[[1, 2, 3]]` returns `[1, 2, 3]` without repeated cells.

## Picture

For `[[1,2,3], [4,5,6]]`, read top `1,2,3`, right `6`, then bottom `5,4`.
The unread bounds have crossed, so stop.

<!-- visual:spiral-matrix-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="spiral-matrix-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="spiral-matrix-state-title">Recheck the remaining boundaries before traversing another edge</p><div class="visual-argument" data-visual-argument="spiral-matrix-state" data-argument-kind="ledger"><p class="visual-argument-premise">Worked example: a crossed boundary prevents revisiting a one-row remainder</p><svg viewBox="0 0 360 322" role="img" aria-label="Worked example: a crossed boundary prevents revisiting a one-row remainder. For a single row (1,2,3), the top traversal consumes every cell. The bottom traversal must be skipped after top passes bottom."><title>Worked example: a crossed boundary prevents revisiting a one-row remainder</title><desc>For a single row (1,2,3), the top traversal consumes every cell. The bottom traversal must be skipped after top passes bottom.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">One-row input</tspan></text><rect data-proof-token="1" x="20" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="71.33333333333334" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">1</tspan></text><rect data-proof-token="2" x="126.66666666666667" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="178" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">2</tspan></text><rect data-proof-token="3" x="233.33333333333334" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="284.6666666666667" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">3</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">After traversing top</tspan></text><rect data-proof-token="top=1" x="20" y="156" width="156" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="98" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">top=1</tspan></text><rect data-proof-token="bottom=0" x="180" y="156" width="156" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="258" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">bottom=0</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Bottom traversal</tspan></text><rect data-proof-token="skip" x="20" y="254" width="316" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="178" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">skip</tspan></text></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="spiral-matrix" role="group" tabindex="0" aria-label="Spiral Matrix: Read one side of the current rectangle at a time, shrink that boundary, and guard sides that may have disappeared."><div class="coding-visual-example"><span>Input and goal</span><strong>Return matrix values in spiral order.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize the outer rectangle"><div class="coding-trace-frame-heading"><span>Initialize the outer rectangle</span><strong>Bounds are top=0, bottom=2, left=0, right=3. The answer starts empty.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="spiral-cursor"><span>1</span><small>read cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>10</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>11</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>12</span></span></div><div class="coding-trace-meta"><span><b>boundaries</b>top=0 bottom=2 left=0 right=3</span><span><b>answer</b>[]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>read cursor</dt><dd><strong class="coding-state-after">(0, 0)</strong></dd></div><div class="coding-state-change"><dt>boundaries</dt><dd><strong class="coding-state-after">top=0 bottom=2 left=0 right=3</strong></dd></div><div class="coding-state-change"><dt>answer</dt><dd><strong class="coding-state-after">[]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="read-top" hidden role="group" aria-label="Read the top edge left to right"><div class="coding-trace-frame-heading"><span>Read the top edge left to right</span><strong>Append row 0 from columns 0 through 3: 1,2,3,4; then increment top to 1.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="spiral-cursor"><span>4</span><small>read cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>10</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>11</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>12</span></span></div><div class="coding-trace-meta"><span><b>direction</b>(0,0) -&gt; (0,3)</span><span><b>boundaries</b>top=1 bottom=2 left=0 right=3</span><span><b>answer</b>[1,2,3,4]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>read cursor</dt><dd><span class="coding-state-before">(0, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 3)</strong></dd></div><div class="coding-state-change"><dt>direction</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0,0) -&gt; (0,3)</strong></dd></div><div class="coding-state-change"><dt>boundaries</dt><dd><span class="coding-state-before">top=0 bottom=2 left=0 right=3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">top=1 bottom=2 left=0 right=3</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="read-right" hidden role="group" aria-label="Read the right edge downward"><div class="coding-trace-frame-heading"><span>Read the right edge downward</span><strong>Append column 3 from rows 1 through 2: 8,12; then decrement right to 2.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>10</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>11</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="spiral-cursor"><span>12</span><small>read cursor</small></span></div><div class="coding-trace-meta"><span><b>direction</b>(1,3) -&gt; (2,3)</span><span><b>boundaries</b>top=1 bottom=2 left=0 right=2</span><span><b>answer</b>[1,2,3,4,8,12]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>read cursor</dt><dd><span class="coding-state-before">(0, 3)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 3)</strong></dd></div><div class="coding-state-change"><dt>direction</dt><dd><span class="coding-state-before">(0,0) -&gt; (0,3)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1,3) -&gt; (2,3)</strong></dd></div><div class="coding-state-change"><dt>boundaries</dt><dd><span class="coding-state-before">top=1 bottom=2 left=0 right=3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">top=1 bottom=2 left=0 right=2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="read-bottom" hidden role="group" aria-label="Read the bottom edge right to left"><div class="coding-trace-frame-heading"><span>Read the bottom edge right to left</span><strong>top&lt;=bottom, so append row 2 from columns 2 through 0 in reverse: 11,10,9; then bottom becomes 1.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>8</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="spiral-cursor"><span>9</span><small>read cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>10</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>11</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>12</span></span></div><div class="coding-trace-meta"><span><b>direction</b>(2,2) -&gt; (2,0)</span><span><b>boundaries</b>top=1 bottom=1 left=0 right=2</span><span><b>answer</b>[1,2,3,4,8,12,11,10,9]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>read cursor</dt><dd><span class="coding-state-before">(2, 3)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 0)</strong></dd></div><div class="coding-state-change"><dt>direction</dt><dd><span class="coding-state-before">(1,3) -&gt; (2,3)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2,2) -&gt; (2,0)</strong></dd></div><div class="coding-state-change"><dt>boundaries</dt><dd><span class="coding-state-before">top=1 bottom=2 left=0 right=2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">top=1 bottom=1 left=0 right=2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="read-left" hidden role="group" aria-label="Read the left edge upward"><div class="coding-trace-frame-heading"><span>Read the left edge upward</span><strong>left&lt;=right, so append column 0 from row 1 up to row 1: 5; then increment left to 1.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>4</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="spiral-cursor"><span>5</span><small>read cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>6</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>10</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>11</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>12</span></span></div><div class="coding-trace-meta"><span><b>direction</b>(1,0) -&gt; (1,0)</span><span><b>boundaries</b>top=1 bottom=1 left=1 right=2</span><span><b>answer</b>[1,2,3,4,8,12,11,10,9,5]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>read cursor</dt><dd><span class="coding-state-before">(2, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 0)</strong></dd></div><div class="coding-state-change"><dt>direction</dt><dd><span class="coding-state-before">(2,2) -&gt; (2,0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1,0) -&gt; (1,0)</strong></dd></div><div class="coding-state-change"><dt>boundaries</dt><dd><span class="coding-state-before">top=1 bottom=1 left=0 right=2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">top=1 bottom=1 left=1 right=2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="read-inner-top" hidden role="group" aria-label="Read the inner top edge"><div class="coding-trace-frame-heading"><span>Read the inner top edge</span><strong>The remaining rectangle is row 1, columns 1..2. Append 6,7 and increment top to 2.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>5</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>6</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="spiral-cursor"><span>7</span><small>read cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>10</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>11</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>12</span></span></div><div class="coding-trace-meta"><span><b>direction</b>(1,1) -&gt; (1,2)</span><span><b>boundaries</b>top=2 bottom=1 left=1 right=2</span><span><b>answer</b>[1,2,3,4,8,12,11,10,9,5,6,7]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>read cursor</dt><dd><span class="coding-state-before">(1, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 2)</strong></dd></div><div class="coding-state-change"><dt>direction</dt><dd><span class="coding-state-before">(1,0) -&gt; (1,0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1,1) -&gt; (1,2)</strong></dd></div><div class="coding-state-change"><dt>boundaries</dt><dd><span class="coding-state-before">top=1 bottom=1 left=1 right=2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">top=2 bottom=1 left=1 right=2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="finish" hidden role="group" aria-label="Skip vanished sides and stop"><div class="coding-trace-frame-heading"><span>Skip vanished sides and stop</span><strong>The right-edge loop is empty and makes right=1; top&gt;bottom skips the bottom, then the while condition fails.</strong></div><div class="coding-trace-grid" style="--trace-cols:4" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>2</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-2"><span>3</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-3"><span>4</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>5</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="spiral-cursor"><span>6</span><small>read cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-2"><span>7</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-3"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>10</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-2"><span>11</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-3"><span>12</span></span></div><div class="coding-trace-meta"><span><b>boundaries</b>top=2 bottom=1 left=2 right=1</span><span><b>guard</b>top&gt;bottom; no row remains</span><span><b>result</b>[1,2,3,4,8,12,11,10,9,5,6,7]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>read cursor</dt><dd><span class="coding-state-before">(1, 2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 1)</strong></dd></div><div class="coding-state-change"><dt>boundaries</dt><dd><span class="coding-state-before">top=2 bottom=1 left=1 right=2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">top=2 bottom=1 left=2 right=1</strong></dd></div><div class="coding-state-change"><dt>guard</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">top&gt;bottom; no row remains</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 7</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize the outer rectangle</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Read the top edge left to right</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read the right edge downward</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Read the bottom edge right to left</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Read the left edge upward</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Read the inner top edge</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Skip vanished sides and stop</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use boundary peeling when a rectangular grid must be traversed layer by layer around its perimeter in directional order.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">At each while-loop entry, every cell outside top..bottom and left..right has been emitted exactly once, and every unvisited cell lies inside that closed rectangle.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For perimeter traversals, represent remaining work as explicit inclusive bounds, consume one edge, shrink it immediately, and guard later edges because earlier moves may collapse the region.</dd></div></dl></div><div class="visual-derivation-caption">Bounds are top=0, bottom=2, left=0, right=3. The answer starts empty. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="spiral-matrix-state"><strong>Check:</strong> Why must the bottom-edge check be skipped after the last inner top row is consumed?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="spiral-matrix-state">For a single row (1,2,3), the top traversal consumes every cell. The bottom traversal must be skipped after top passes bottom.</span></figcaption></figure>

Each completed side is removed from the unread rectangle before another side is visited.
The bottom and left guards prevent revisiting cells after a thin rectangle disappears.

## Code

```python
def spiral_order(matrix: list[list[int]]) -> list[int]:
   if not matrix or not matrix[0]:
      return []

   answer = []
   top, bottom = 0, len(matrix) - 1
   left, right = 0, len(matrix[0]) - 1

   while top <= bottom and left <= right:
      answer.extend(matrix[top][left : right + 1])
      top += 1

      for row in range(top, bottom + 1):
         answer.append(matrix[row][right])
      right -= 1

      if top <= bottom:
         answer.extend(reversed(matrix[bottom][left : right + 1]))
         bottom -= 1
      if left <= right:
         for row in range(bottom, top - 1, -1):
            answer.append(matrix[row][left])
         left += 1

   return answer
```

**Cost:** $O(rows\times cols)$ cell visits and output storage, plus $O(cols)$ temporary slicing space.
The four bounds themselves use constant state.

## Change the Requirement

### Stream Coordinates Without Slices

**New task:** Yield positions lazily using constant auxiliary state.

**Picture and change:** Keep the rectangle but replace copied rows with indexed visits.

**Code:** Nonnegative integer dimensions are required.

```python
def spiral_positions(rows: int, cols: int):
   if rows < 0 or cols < 0:
      raise ValueError("dimensions must be nonnegative")
   top, bottom, left, right = 0, rows - 1, 0, cols - 1
   while top <= bottom and left <= right:
      for col in range(left, right + 1):
         yield top, col
      top += 1
      for row in range(top, bottom + 1):
         yield row, right
      right -= 1
      if top <= bottom:
         for col in range(right, left - 1, -1):
            yield bottom, col
         bottom -= 1
      if left <= right:
         for row in range(bottom, top - 1, -1):
            yield row, left
         left += 1
```

**Check:** `list(spiral_positions(1, 3))` returns `[(0,0), (0,1), (0,2)]` once each.

**Use when:** Consumers need positions incrementally.
Exhaustion costs $O(rows\times cols)$ time; storing yielded positions is the consumer's choice.

### Reverse Traversal Orientation

**New task:** Visit counterclockwise, still starting at top-left.

**Picture and change:** Transposition exchanges horizontal and vertical movements, reversing orientation.

**Code:** Reuse the complete reference on a transposed copy.

```python
def spiral_counterclockwise(matrix: list[list[int]]) -> list[int]:
   if not matrix or not matrix[0]:
      return []
   transposed = [
      [matrix[row][col] for row in range(len(matrix))]
      for col in range(len(matrix[0]))
   ]
   return spiral_order(transposed)
```

**Check:** `[[1,2,3], [4,5,6]]` produces `[1,4,5,6,3,2]`.

**Use when:** The initial direction is down rather than right.
The copy adds $O(rows\times cols)$ workspace.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

What happens without the bottom-side guard?

<details>
<summary>Reasoned answer</summary>

For `[[1,2,3]]`, consuming the top row leaves `top > bottom`.
Reading the bottom anyway would append previously visited cells.
The guard checks that a side still belongs to the unread rectangle.

</details>

</div>
