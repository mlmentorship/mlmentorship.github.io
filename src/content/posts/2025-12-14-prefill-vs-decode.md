---
title: "Prefill vs. decode: the two phases of LLM inference"
description: "LLM inference has two cost regimes with very different bottlenecks. Mixing them up leads to wrong cost models and bad serving decisions."
date: "2025-12-14"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Prefill processes available prompt positions together; decode processes a newly sampled input after its predecessor becomes available.
Compare available rows per weight read before naming the bottleneck.

Consider a $1024\times1024$ BF16 weight matrix W: $2{,}097{,}152$ bytes, or 2 MiB.
Assume each invocation streams W once, with no inter-call residency.
Ignore input/output bytes, KV traffic and kernel overhead for this accounting exercise.

How do four simultaneous rows differ from four serial one-row invocations?

<!-- visual:prefill-decode-weight-reuse -->
<figure class="learning-figure" aria-labelledby="prefill-decode-visual-title" aria-describedby="prefill-decode-visual-description">
	<p class="visual-kicker">Same arithmetic; different ideal weight traffic</p>
	<p class="visual-title" id="prefill-decode-visual-title">Shared weight loads can serve the same arithmetic across several tokens</p><div class="visual-argument" data-visual-argument="prefill-decode-weight-reuse" data-argument-kind="ledger"><p class="visual-argument-premise">One weight load can serve more than one token computation</p><svg viewBox="0 0 360 322" role="img" aria-label="One weight load can serve more than one token computation. Prefill or batching can reuse loaded weights across tokens. Decode still performs token-specific computation and reads its growing cache."><title>One weight load can serve more than one token computation</title><desc>Prefill or batching can reuse loaded weights across tokens. Decode still performs token-specific computation and reads its growing cache.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">One loaded weight tile</tspan></text><rect data-proof-token="W" x="20" y="58" width="316" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="178" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">W</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Several token computations reuse W</tspan></text><rect data-proof-token="token 1" x="20" y="156" width="76" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="58" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">token 1</tspan></text><rect data-proof-token="token 2" x="100" y="156" width="76" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="138" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">token 2</tspan></text><rect data-proof-token="token 3" x="180" y="156" width="76" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="218" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">token 3</tspan></text><rect data-proof-token="token 4" x="260" y="156" width="76" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="298" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">token 4</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Not removed by weight reuse</tspan></text><rect data-proof-token="token math" x="20" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="98" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">token math</tspan></text><rect data-proof-token="KV reads" x="180" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="258" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">KV reads</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<p id="prefill-decode-visual-description">W has shape 1024 by 1024. The serial calls represent four subsequent decode steps, not four total output tokens.</p>
	<div class="visual-grid--two" role="group" aria-label="Prefill and decode weight-reuse comparison">
		<section class="visual-panel" aria-labelledby="prefill-weight-reuse-title">
			<h4 id="prefill-weight-reuse-title">Prefill · one parallel pass</h4>
			<p>Four prompt-token rows are available together.</p>
			<table class="cm-grid" aria-label="One prefill weight tile reused across four prompt rows">
				<thead><tr><th scope="col">HBM supplies</th><th scope="col">Matrix work</th></tr></thead>
				<tbody><tr><td><strong>weight tile W</strong><br>load for this tile</td><td class="cm-selected"><strong>rows 1 · 2 · 3 · 4</strong><br>reuse W across prompt rows</td></tr></tbody>
			</table>
			<p><strong>Weight-only accounting:</strong> 8,388,608 FLOPs / 2,097,152 bytes = 4 FLOPs/byte.</p>
		</section>
		<section class="visual-panel" aria-labelledby="decode-weight-stream-title">
			<h4 id="decode-weight-stream-title">Decode · four serial passes</h4>
			<p>At batch 1, only one new-token row exists per step.</p>
			<table class="cm-grid" aria-label="The same decode weight tile streamed again for each serial output step">
				<thead><tr><th scope="col">Decode invocation</th><th scope="col">HBM and matrix work</th></tr></thead>
				<tbody>
					<tr><th scope="row">1</th><td>load W → row 1</td></tr>
					<tr><th scope="row">2</th><td>load W again → row 2</td></tr>
					<tr><th scope="row">3</th><td>load W again → row 3</td></tr>
					<tr><th scope="row">4</th><td>load W again → row 4</td></tr>
				</tbody>
			</table>
			<p><strong>Weight-only accounting:</strong> 8,388,608 FLOPs / 8,388,608 bytes = 1 FLOP/byte.</p>
		</section>
	</div>
	<div class="visual-derivation-caption">four parallel rows reuse each loaded weight. Serial one-row calls repeat the stream under the stated residency assumption; actual traffic requires measurement.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="prefill-decode-weight-reuse"><strong>Check:</strong> Why is arithmetic identical while the illustrated serial path loads four times as many weight bytes?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="prefill-decode-weight-reuse">Prefill or batching can reuse loaded weights across tokens. Decode still performs token-specific computation and reads its growing cache.</span></figcaption>
</figure>

One row costs $2(1024)^2=2{,}097{,}152$ FLOPs, counting a multiply-add as two operations.
Both schedules perform $8{,}388{,}608$ FLOPs; their ideal weight traffic differs fourfold.
Neither ratio alone establishes a hardware bottleneck.

## Prefill

For prompt length $P$, compute K/V and hidden outputs for available positions with the required causal mask.
The final prompt position supplies the first output-token distribution.

Using $N$ participating dense weights, the dense work is approximately $2PN$ FLOPs.
Add attention work $O(LP^2d)$ for $L$ layers and hidden width $d$.
More rows can improve reuse; they need not produce square matrices or saturate the device.

Large efficient shapes can be compute-bound; short prompts can be memory-bound or launch-bound.

## Decode

Each subsequent invocation processes the sampled token, appends its K/V, and predicts the next token.
For batch size $B$, dense work is approximately $2BN$, plus context-dependent attention and KV traffic.

Small-batch weight streaming can dominate, but long-context KV reads, compute and launches can become limiting.
See the [KV state ledger](/concepts/kv-cache/) and [resource lower bounds](/concepts/gpu-memory-hierarchy/).

## Latency metrics

TTFT measures arrival-to-first-token latency, including queueing, scheduling, prompt processing, sampling and transport as applicable.
Inter-token intervals describe later outputs; TPOT is often their average, not a constant service time.

For $T\ge1$ outputs, end-to-end latency is TTFT plus the $T-1$ subsequent intervals.
A separate hypothetical request with TTFT 100 ms and three 20-ms intervals finishes four outputs in $100+3(20)=160$ ms.

## Change the rows or arrival clock

### 1. Sixteen requests are ready

**Picture and change:** batch sixteen independent decode rows while retaining separate positions, masks, KV and sampler states.

**Check:** weight-only intensity becomes 16 FLOPs/byte, not a guaranteed sixteenfold speedup.

**Use when:** measurements identify weight traffic as a limiting term.

### 2. Add 20 ms of queueing

**Picture and change:** measure from arrival, not GPU launch.
If prefills block decoding, bound chunks and track the prompt cursor with KV.

**Check:** TTFT becomes 120 ms and four-output latency 180 ms, with other timings unchanged.

**Use when:** load or long prompts violate first-token or inter-token targets.

## Failure and transfer

Calling every prefill compute-bound misses launch-bound short prompts; calling every decode weight-bound misses other traffic and compute.

How many subsequent decode invocations normally produce four outputs after prefill supplies the first?

<details>
<summary>Check your answer</summary>

Three. The first sampled output enters KV only when processed by a subsequent invocation, not when sampled.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## What follows from the asymmetry

| Optimization | Conditional effect, not a universal phase ranking |
| --- | --- |
| Larger batches | Reuse weights across rows; queueing, compute and KV can limit benefits |
| FlashAttention | Reduce attention traffic; actual gain depends on shape and implementation |
| Weight quantization | Reduce weight bytes; kernel support, conversion and errors matter |
| KV quantization | Change cache capacity/traffic, including writes; metadata and accuracy remain costs |
| Speculation | Spend draft/verification work to reduce sequential target calls when accepted |
| GQA/MQA | Change architecture and KV size, requiring compatible trained weights |
| Continuous batching | Reuse completed request positions under token, memory and fairness constraints |

## Common pitfalls

Measure full workloads, not a single inference-cost number.
Neither output length alone nor phase name determines the dominant latency term.
Speculation can lose on short outputs when overhead exceeds saved work.

References: [Pope et al.](https://arxiv.org/abs/2211.05102), [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), [SARATHI](https://arxiv.org/abs/2308.16369).

## Related

[Continuous batching](/concepts/continuous-batching/) covers admission and per-request state.
[Speculative decoding](/concepts/speculative-decoding/) covers draft verification.

</div>
