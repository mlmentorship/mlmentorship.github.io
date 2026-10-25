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
	<p class="visual-title" id="continuous-batching-slot-title">Rebuilding membership fills three otherwise idle decode positions.</p>
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
	<figcaption><strong>Read it this way:</strong> both schedules finish B after iteration 2. Continuous membership admits ready C at iteration 3; static membership waits for A to finish.</figcaption>
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
