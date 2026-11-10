---
title: "Continuous batching for LLM serving"
description: "Reuse finished request positions between iterations while preserving per-request state and respecting token and KV budgets."
date: "2026-04-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Continuous batching rebuilds the active request set between model iterations, allowing queued work to replace a completed request.

Consider two decode positions: A needs five more iterations, B two, and queued C three.
C's prefill is complete; all required KV reservations fit.
Each occupied cell below represents one scheduled decode iteration, not prompt processing.

When can C use B's position without disturbing A's ordered state?

<!-- visual:continuous-batching-reuses-finished-slot -->
<figure class="learning-figure" aria-labelledby="continuous-batching-slot-title">
	<p class="visual-kicker">B ends after iteration 2; C is ready</p>
	<p class="visual-title" id="continuous-batching-slot-title">Reuse finished slots only when prompt readiness and resource budgets permit</p><div class="visual-argument" data-visual-argument="continuous-batching-reuses-finished-slot" data-argument-kind="ledger" data-argument-composition="slot-reuse-and-admission"><svg viewBox="0 0 360 1540" role="img" aria-label="Reuse finished slots only when prompt readiness and resource budgets permit. At iteration boundaries, keep continuing requests' state and reuse finished slots. The top case assumes C is ready and fits. With an eight-token prompt and two tokens left per iteration, C needs four prefill chunks before decoding. Three required KV blocks cannot fit into two free blocks, even with an empty row. Admission must satisfy readiness, token work, and KV limits."><title>Reuse finished slots only when prompt readiness and resource budgets permit</title><desc>At iteration boundaries, keep continuing requests' state and reuse finished slots. The top case assumes C is ready and fits. With an eight-token prompt and two tokens left per iteration, C needs four prefill chunks before decoding. Three required KV blocks cannot fit into two free blocks, even with an empty row. Admission must satisfy readiness, token work, and KV limits.</desc><text x="20" y="26" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A keeps its slot while B leaves</tspan></text><text x="20" y="68" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">This iteration</tspan></text><rect data-picture-slot="0,0" data-picture-job="A" x="25" y="88" width="125" height="59" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><text x="87.5" y="126" style="font-size:26px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="87.5" dy="0">A</tspan></text><rect data-picture-slot="0,1" data-picture-job="B" x="210" y="88" width="125" height="59" rx="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="272.5" y="126" style="font-size:26px;font-weight:700;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="272.5" dy="0">B</tspan></text><text x="130" y="294" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="130" dy="0">Next iteration</tspan></text><rect data-picture-slot="1,0" data-picture-job="A" x="25" y="314" width="125" height="59" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><text x="87.5" y="352" style="font-size:26px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="87.5" dy="0">A</tspan></text><rect data-picture-slot="1,1" data-picture-job="C" x="210" y="314" width="125" height="59" rx="4" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="272.5" y="352" style="font-size:26px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="272.5" dy="0">C</tspan></text><path d="M87.5 159L87.5 297" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M82.5 289L87.5 297" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M92.5 289L87.5 297" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="111" y="218" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="111" dy="0">still</tspan><tspan x="111" dy="18">running</tspan></text><path d="M272.5 159Q272.5 192 320 210" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path><circle cx="320" cy="210" r="18" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><text x="320" y="215" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="320" dy="0">B</tspan></text><text x="245" y="184" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="245" dy="0">done</tspan></text><circle cx="218" cy="237" r="18" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="218" y="242" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="218" dy="0">C</tspan></text><path d="M230 251L272.5 297" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M265 294L272.5 297" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M272.5 289L272.5 297" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="20" y="421" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A stays. The freed slot receives C.</tspan></text><text x="20" y="521" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Top: C is ready and fits. A retains</tspan><tspan x="20" dy="18">its KV and sampler state.</tspan></text><text x="20" y="612" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Changed condition: C still needs</tspan><tspan x="20" dy="18">prefill</tspan></text><text x="20" y="670" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Budget 4; ongoing decode uses 2;</tspan><tspan x="20" dy="18">prefill gets 2</tspan></text><text x="20" y="727" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Chunk 1: unfinished</tspan></text><rect x="20" y="744" width="40" height="35" data-picture-batch-prefill-progress="2" data-picture-batch-prefill-token="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="60" y="744" width="40" height="35" data-picture-batch-prefill-progress="2" data-picture-batch-prefill-token="1" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="100" y="744" width="40" height="35" data-picture-batch-prefill-progress="2" data-picture-batch-prefill-token="2" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="140" y="744" width="40" height="35" data-picture-batch-prefill-progress="2" data-picture-batch-prefill-token="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="180" y="744" width="40" height="35" data-picture-batch-prefill-progress="2" data-picture-batch-prefill-token="4" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="220" y="744" width="40" height="35" data-picture-batch-prefill-progress="2" data-picture-batch-prefill-token="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="260" y="744" width="40" height="35" data-picture-batch-prefill-progress="2" data-picture-batch-prefill-token="6" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="300" y="744" width="40" height="35" data-picture-batch-prefill-progress="2" data-picture-batch-prefill-token="7" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="20" y="817" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Chunk 2: unfinished</tspan></text><rect x="20" y="834" width="40" height="35" data-picture-batch-prefill-progress="4" data-picture-batch-prefill-token="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="60" y="834" width="40" height="35" data-picture-batch-prefill-progress="4" data-picture-batch-prefill-token="1" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="100" y="834" width="40" height="35" data-picture-batch-prefill-progress="4" data-picture-batch-prefill-token="2" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="140" y="834" width="40" height="35" data-picture-batch-prefill-progress="4" data-picture-batch-prefill-token="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="180" y="834" width="40" height="35" data-picture-batch-prefill-progress="4" data-picture-batch-prefill-token="4" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="220" y="834" width="40" height="35" data-picture-batch-prefill-progress="4" data-picture-batch-prefill-token="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="260" y="834" width="40" height="35" data-picture-batch-prefill-progress="4" data-picture-batch-prefill-token="6" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="300" y="834" width="40" height="35" data-picture-batch-prefill-progress="4" data-picture-batch-prefill-token="7" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="20" y="907" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Chunk 3: unfinished</tspan></text><rect x="20" y="924" width="40" height="35" data-picture-batch-prefill-progress="6" data-picture-batch-prefill-token="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="60" y="924" width="40" height="35" data-picture-batch-prefill-progress="6" data-picture-batch-prefill-token="1" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="100" y="924" width="40" height="35" data-picture-batch-prefill-progress="6" data-picture-batch-prefill-token="2" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="140" y="924" width="40" height="35" data-picture-batch-prefill-progress="6" data-picture-batch-prefill-token="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="180" y="924" width="40" height="35" data-picture-batch-prefill-progress="6" data-picture-batch-prefill-token="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="220" y="924" width="40" height="35" data-picture-batch-prefill-progress="6" data-picture-batch-prefill-token="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="260" y="924" width="40" height="35" data-picture-batch-prefill-progress="6" data-picture-batch-prefill-token="6" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="300" y="924" width="40" height="35" data-picture-batch-prefill-progress="6" data-picture-batch-prefill-token="7" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="20" y="997" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Chunk 4: ready</tspan></text><rect x="20" y="1014" width="40" height="35" data-picture-batch-prefill-progress="8" data-picture-batch-prefill-token="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="60" y="1014" width="40" height="35" data-picture-batch-prefill-progress="8" data-picture-batch-prefill-token="1" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="100" y="1014" width="40" height="35" data-picture-batch-prefill-progress="8" data-picture-batch-prefill-token="2" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="140" y="1014" width="40" height="35" data-picture-batch-prefill-progress="8" data-picture-batch-prefill-token="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="180" y="1014" width="40" height="35" data-picture-batch-prefill-progress="8" data-picture-batch-prefill-token="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="220" y="1014" width="40" height="35" data-picture-batch-prefill-progress="8" data-picture-batch-prefill-token="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="260" y="1014" width="40" height="35" data-picture-batch-prefill-progress="8" data-picture-batch-prefill-token="6" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="300" y="1014" width="40" height="35" data-picture-batch-prefill-progress="8" data-picture-batch-prefill-token="7" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1110" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">C can decode only after the final</tspan><tspan x="20" dy="18">prefill chunk.</tspan></text><text x="20" y="1210" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Changed condition: an empty row lacks</tspan><tspan x="20" dy="18">KV</tspan></text><text x="20" y="1290" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Free: 2 blocks</tspan></text><text x="185" y="1290" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="185" dy="0">Need: 3 blocks</tspan></text><rect x="30" y="1320" width="52" height="48" data-picture-batch-free-block="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect x="92" y="1320" width="52" height="48" data-picture-batch-free-block="1" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect x="185" y="1320" width="48" height="48" data-picture-batch-required-block="0" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><rect x="237" y="1320" width="48" height="48" data-picture-batch-required-block="1" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><rect x="289" y="1320" width="48" height="48" data-picture-batch-required-block="2" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="20" y="1440" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">The third block has no available</tspan><tspan x="20" dy="18">capacity. Defer admission.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Static and continuous schedules for the same five decode steps">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 210" role="img" aria-labelledby="static-batch-title static-batch-desc">
				<title id="static-batch-title">Static batching leaves a finished request's slot idle</title>
				<desc id="static-batch-desc">Across five decode steps, request A occupies slot one for all five steps. Request B occupies slot two for steps one and two, then slot two is idle for steps three through five. Queued request C waits because the static batch cannot change until A also finishes. Seven of ten slot positions do useful work.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="177" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">STATIC · MEMBERSHIP FIXED</text>
				<text class="viz-axis-label" x="90" y="47" text-anchor="middle">1</text>
				<text class="viz-axis-label" x="128" y="47" text-anchor="middle">2</text>
				<text class="viz-axis-label" x="166" y="47" text-anchor="middle">3</text>
				<text class="viz-axis-label" x="204" y="47" text-anchor="middle">4</text>
				<text class="viz-axis-label" x="242" y="47" text-anchor="middle">5</text>
				<text class="viz-label" x="14" y="78">slot 1</text>
				<text class="viz-label" x="14" y="118">slot 2</text>
				<rect class="viz-node viz-node--input" x="75" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="113" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="151" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="189" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="227" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node" x="75" y="100" width="30" height="28" rx="3"></rect>
				<rect class="viz-node" x="113" y="100" width="30" height="28" rx="3"></rect>
				<rect x="151" y="100" width="30" height="28" rx="3" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:3 2"></rect>
				<rect x="189" y="100" width="30" height="28" rx="3" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:3 2"></rect>
				<rect x="227" y="100" width="30" height="28" rx="3" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:3 2"></rect>
				<text class="viz-node-label" x="90" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="128" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="166" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="204" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="242" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="90" y="119" text-anchor="middle">B</text>
				<text class="viz-node-label" x="128" y="119" text-anchor="middle">B</text>
				<text class="viz-label" x="166" y="118" text-anchor="middle">idle</text>
				<text class="viz-label" x="204" y="118" text-anchor="middle">idle</text>
				<text class="viz-label" x="242" y="118" text-anchor="middle">idle</text>
				<text class="viz-callout" x="14" y="156">C waits until A finishes after step 5.</text>
				<text class="viz-axis-label" x="14" y="185">USEFUL SLOTS: 7 / 10</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 210" role="img" aria-labelledby="continuous-batch-title continuous-batch-desc">
				<title id="continuous-batch-title">Continuous batching replaces B with queued request C at the next iteration</title>
				<desc id="continuous-batch-desc">Across the same five decode steps, request A occupies slot one throughout. Request B occupies slot two for steps one and two. After B completes, the scheduler admits queued request C into slot two for steps three through five while A continues. All ten slot positions do useful work.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="177" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">CONTINUOUS · REBUILD EACH STEP</text>
				<text class="viz-axis-label" x="90" y="47" text-anchor="middle">1</text>
				<text class="viz-axis-label" x="128" y="47" text-anchor="middle">2</text>
				<text class="viz-axis-label" x="166" y="47" text-anchor="middle">3</text>
				<text class="viz-axis-label" x="204" y="47" text-anchor="middle">4</text>
				<text class="viz-axis-label" x="242" y="47" text-anchor="middle">5</text>
				<text class="viz-label" x="14" y="78">slot 1</text>
				<text class="viz-label" x="14" y="118">slot 2</text>
				<rect class="viz-node viz-node--input" x="75" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="113" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="151" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="189" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--input" x="227" y="60" width="30" height="28" rx="3"></rect>
				<rect class="viz-node" x="75" y="100" width="30" height="28" rx="3"></rect>
				<rect class="viz-node" x="113" y="100" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--output" x="151" y="100" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--output" x="189" y="100" width="30" height="28" rx="3"></rect>
				<rect class="viz-node viz-node--output" x="227" y="100" width="30" height="28" rx="3"></rect>
				<text class="viz-node-label" x="90" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="128" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="166" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="204" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="242" y="79" text-anchor="middle">A</text>
				<text class="viz-node-label" x="90" y="119" text-anchor="middle">B</text>
				<text class="viz-node-label" x="128" y="119" text-anchor="middle">B</text>
				<text class="viz-node-label" x="166" y="119" text-anchor="middle">C</text>
				<text class="viz-node-label" x="204" y="119" text-anchor="middle">C</text>
				<text class="viz-node-label" x="242" y="119" text-anchor="middle">C</text>
				<path d="M147 137L155 145L163 137" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<text class="viz-callout" x="160" y="153">C admitted for step 3</text>
				<text class="viz-axis-label" x="14" y="185">USEFUL SLOTS: 10 / 10</text>
			</svg>
		</section>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> both schedules finish B after iteration 2. Continuous membership admits ready C at iteration 3; static membership waits for A to finish.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="continuous-batching-reuses-finished-slot"><strong>Check:</strong> Which exact cells change from idle to C? When can C first start?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="continuous-batching-reuses-finished-slot">At iteration boundaries, keep continuing requests' state and reuse finished slots. The top case assumes C is ready and fits. With an eight-token prompt and two tokens left per iteration, C needs four prefill chunks before decoding. Three required KV blocks cannot fit into two free blocks, even with an empty row. Admission must satisfy readiness, token work, and KV limits. <strong>Must A finish before C can occupy B’s released slot?</strong></span></figcaption>
</figure>

Static membership uses $5+2=7$ of ten positions: 70% occupancy.
Continuous membership uses $5+2+3=10$: 100% occupancy, with C finished after iteration 5.
This is not a measured $10/7$ throughput gain: iteration durations can differ.

## The mechanism

Maintain a queue and request-state ledger keyed by request ID, not batch row.

| Request | Phase | Remaining iterations | Owned state |
| --- | --- | --- | --- |
| A | Decode | $5$ | Valid KV length, positions, stop count, RNG |
| B | Decode | $2$ | Same fields, separately owned |
| C | Ready/queued | $3$ | Completed prefill and reserved capacity |

1. Select decode tokens and eligible prefill chunks within token, memory and fairness limits.
2. Run the supported batch; update each request's valid KV, position and sampling state.
3. On EOS, `max_tokens` or cancellation, release that request's owned references safely.
4. Rebuild membership for the next iteration; preserve each continuing request's identity and order.

Packed offsets such as `cu_seqlens` can isolate variable-length sequences.
Paging is one cache-management choice, not a requirement of continuous scheduling.

## Change the prefill or capacity constraint

### 1. C's eight-token prompt is unfinished

**Picture and change:** permit prefill alongside two continuing decoders under a four-token iteration budget.
Reserve two decode tokens, leaving at most two prompt tokens per iteration.

**Check:** four two-token chunks finish the prompt, assuming both decoders remain active.
Track the prompt cursor and existing KV together.
C cannot use first-token logits before its final prefill chunk completes.

**Use when:** long prefills threaten existing requests' inter-token latency.

### 2. A free row lacks KV capacity

**Picture and change:** C needs three additional KV blocks, but only two are free.
Defer admission or use an explicitly supported preemption policy.

**Check:** an empty row does not authorize allocation; cancellation must release only C's references.

**Use when:** context lengths make row counts a poor memory proxy.

## Failure and transfer

Moving A to another row while retaining row-indexed KV or sampler state can mix requests.
A global RNG can also change A's sampled string after C joins.

B finishes, but C's prompt is unprocessed. Can C immediately decode?

<details>
<summary>Check your answer</summary>

No. Schedule prefill within both budgets; completed prefill supplies initial KV and the first-token distribution.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Prefill vs. decode

Prefill processes prompt positions; cached decode appends to per-request state.
Small-batch decode can amortize weight traffic across requests, but bottlenecks depend on workload and hardware.
See [prefill versus decode](/concepts/prefill-vs-decode/) and [GPU memory bounds](/concepts/gpu-memory-hierarchy/).

## Tradeoffs

Batch size affects latency, compute and KV traffic; it does not guarantee throughput improvements.
Admission, queue age and preemption policies determine fairness under memory pressure.
GQA/MQA change architecture and cache size; paging addresses allocation, not all possible bottlenecks.

## Common pitfalls

Measure arrival/length distributions, scheduler overhead, latency targets and memory capacity.
A single-request profile does not establish serving throughput.
Distinguish concurrency from context length; idle positions do not mean the whole GPU is idle.

References: [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), [SARATHI](https://arxiv.org/abs/2308.16369), [PagedAttention](https://arxiv.org/abs/2309.06180).

</div>
