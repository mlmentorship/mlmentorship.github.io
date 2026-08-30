---
title: "Continuous batching for LLM serving"
description: "Let new requests join an in-flight batch at every decode step instead of waiting for the slowest one. The other half of why vLLM is fast."
date: "2026-04-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Continuous batching (a.k.a. iteration-level scheduling) processes a batch one decode step at a time and lets new requests enter the batch as soon as another request finishes, instead of waiting for the entire static batch to complete.

Small-batch LLM decoding is usually memory-bound because each step reads model weights for little arithmetic. Adding requests can amortize those weight reads and improve throughput. Iteration time eventually grows as matrix compute, KV-cache traffic, or scheduler overhead becomes limiting.

With **static batching**, you wait for the longest request in the batch before reusing GPU. If one request generates 1000 tokens and another generates 50, the second request's GPU slot sits idle for 950 steps.

Continuous batching keeps more useful work on the GPU. Combined with paged KV allocation, it is a common design in modern LLM servers. The gain over static batching depends on arrival rate, length distribution, memory capacity, scheduler policy, and latency target.

<!-- visual:continuous-batching-reuses-finished-slot -->
<figure class="learning-figure" aria-labelledby="continuous-batching-slot-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="continuous-batching-slot-title">What changes when the scheduler can rebuild the batch after every decode step?</p>
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
				<path d="M147 137L155 145L163 137" style="fill:none;stroke:var(--viz-edge-strong);stroke-width:2"></path>
				<text class="viz-callout" x="169" y="153">C admitted for step 3</text>
				<text class="viz-axis-label" x="14" y="185">USEFUL SLOTS: 10 / 10</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> B finishes after step 2 in both schedules. Static batching strands that slot until A finishes; continuous batching rebuilds the active set and admits queued C for step 3 while A keeps decoding.</figcaption>
</figure>

## The mechanism

Each "step" of the server runs one forward pass for the current batch:

1. Maintain a queue of pending requests and a set of active (in-flight) requests.
2. Each step, build a batch of (a) one decode token from each active request whose KV cache exists and (b) prefill tokens for newly admitted requests.
3. Run one forward pass. Update each active request's KV cache.
4. For requests that hit EOS or `max_tokens`, mark complete and free their KV blocks.
5. Admit new requests from the queue if there is enough free KV-cache capacity.

This requires the attention kernel to handle variable per-request lengths in the same batch (cu_seqlens-style cumulative offsets) and a non-contiguous KV cache (PagedAttention).

## Prefill vs. decode

Two phases with very different cost profiles:

- **Prefill**: process the full prompt in one parallel matmul. Compute-bound; high arithmetic intensity.
- **Decode**: one new token per step per request. Memory-bound; benefits hugely from batching.

Most servers either alternate prefill and decode steps or interleave (chunked prefill, [Patel et al., 2023](https://arxiv.org/abs/2308.16369)) so neither phase starves the other.

## Tradeoffs

- **Throughput vs. latency**: larger batches mean higher tokens/sec across the server but slightly higher per-request latency. SLO-aware servers cap batch size or fragment large prefills.
- **Memory pressure**: continuous batching is throughput-limited by KV-cache memory, not by compute. PagedAttention removes most fragmentation; GQA / MQA shrink per-request cache.
- **Fairness**: a long-context request consumes more KV per step. Without admission control, it can starve short requests.

## Common pitfalls

- **Profiling decode without batching.** Single-request decode benchmarks dramatically understate server throughput.
- **Confusing batch size with sequence length.** Batch size grows the number of concurrent *requests*; longer sequences grow per-request KV.
- **Assuming static batching is fine for production.** It almost never is. The GPU sits idle whenever any request finishes.
