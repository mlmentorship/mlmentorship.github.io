---
title: "Accelerator network topology for distributed ML"
description: "Place each parallelism axis on hardware links that can carry its message size, frequency, and latency needs."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Accelerator topology describes how GPUs or TPUs are connected and how much bandwidth and latency each connection provides. The topology limits which distributed ML layouts run efficiently.

## Why AI labs care

Two clusters with the same number of accelerators can have very different training speed.

The result depends on:

- fast memory inside each accelerator;
- links among accelerators in one fast domain;
- links among nodes or slices;
- network switches and oversubscription;
- message size and collective algorithm;
- whether communication overlaps with compute.

A parallelism plan must follow this hierarchy.

## The connection hierarchy

### Inside one accelerator

Registers and on-chip memory are small and fast. HBM is larger and slower. Kernels move tiles between these levels.

This path limits memory-bound work such as single-token decode and some elementwise operations.

### Inside a fast accelerator domain

GPU systems often connect a small set of GPUs with NVLink and NVSwitch. TPU systems connect chips with inter-chip links, often arranged as a two- or three-dimensional torus.

These links carry frequent tensor-parallel and collective traffic.

### Across nodes or slices

GPU nodes commonly use InfiniBand or Ethernet. TPU slices use data-center networking when traffic leaves the direct inter-chip domain.

These links usually have lower effective bandwidth or higher latency than local accelerator links.

### Between host and accelerator

PCIe or a CPU-to-accelerator link carries input data, checkpoints, and offloaded state. It is usually much slower than HBM.

## Four measures

### Bandwidth

Bandwidth is bytes transferred per second. It sets the time for a large message.

### Latency

Latency is fixed time before useful transfer completes. It dominates small messages and collectives with many steps.

### Bisection bandwidth

Bisection bandwidth is the smallest total bandwidth across an even split of the network. It shows whether many devices can communicate across the cluster at once.

### Effective collective bandwidth

Peak link bandwidth is not the same as measured collective bandwidth. Protocol overhead, message size, routing, congestion, and library choices reduce the useful value.

Use measurements from the target cluster when available.

## GPU and TPU layouts

### GPU clusters

A GPU node usually forms one fast switched domain. Larger clusters connect nodes through another network level.

This creates a strong placement rule: traffic inside a node is often cheaper than the same traffic across nodes.

### TPU clusters

A TPU slice often uses neighbor links in a torus. Uniform ring-like collectives can use these links efficiently. Point-to-point routes may cross several hops.

The logical device mesh should map important communication axes onto useful physical directions.

Neither design is always faster. The workload and collective pattern determine the result.

## Place parallelism on the topology

| Parallelism | Main traffic | Common placement goal |
| --- | --- | --- |
| Tensor parallel | activations every layer | fastest links and low latency |
| Expert parallel | routed token activations | high all-to-all bandwidth, often a small domain |
| FSDP | parameter all-gather and gradient reduce-scatter | enough batch compute to hide traffic |
| Data parallel | gradient reduction | outer axis when batch size can hide the reduction |
| Pipeline parallel | activations between adjacent stages | useful across slower links because messages are smaller |
| Context parallel | key/value or attention state | topology that supports repeated ring or all-to-all traffic |

These are starting points. Measure the actual model and cluster.

## Large and small messages

A ring collective often reaches high bandwidth for large buffers. A tree may finish a small reduction with fewer steps.

For one collective, estimate:

$$
T_{\text{comm}} \approx T_{\text{latency}} + \frac{\text{message bytes}}{\text{effective bandwidth}}.
$$

The latency term depends on the algorithm and number of communication steps.

Small activation collectives during low-batch inference can be latency-bound even when the link has high peak bandwidth.

## Hierarchical collectives

A large GPU cluster can reduce data in stages:

1. combine values inside each node;
2. combine node results across the scale-out network;
3. distribute the final result inside each node.

**Learning objective:** Trace why only one partial result per fast local group needs to participate in the slower scale-out step.

<!-- visual:hierarchical-all-reduce-path -->
<figure class="learning-figure" aria-labelledby="hierarchical-sum-title">
	<p class="visual-title" id="hierarchical-sum-title">Sum locally, combine the partials, return 10 to every rank</p><div class="visual-argument" data-visual-argument="hierarchical-all-reduce-path" data-argument-kind="ledger"><p class="visual-argument-premise">Combine local partials, then return the same total everywhere</p><svg viewBox="0 0 360 322" role="img" aria-label="Combine local partials, then return the same total everywhere. The local sums are 1+3=4 and 2+4=6. Their global sum is 10, which every rank receives."><title>Combine local partials, then return the same total everywhere</title><desc>The local sums are 1+3=4 and 2+4=6. Their global sum is 10, which every rank receives.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Four rank-local inputs</tspan></text><rect data-proof-token="1" x="20" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="58" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">1</tspan></text><rect data-proof-token="3" x="100" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="138" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">3</tspan></text><rect data-proof-token="2" x="180" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="218" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">2</tspan></text><rect data-proof-token="4" x="260" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="298" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">4</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Two local sums cross the hierarchy</tspan></text><rect data-proof-token="4" x="20" y="156" width="156" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="98" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">4</tspan></text><rect data-proof-token="6" x="180" y="156" width="156" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="258" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">6</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Every rank after all-reduce</tspan></text><rect data-proof-token="10" x="20" y="254" width="76" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="58" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">10</tspan></text><rect data-proof-token="10" x="100" y="254" width="76" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="138" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">10</tspan></text><rect data-proof-token="10" x="180" y="254" width="76" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="218" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">10</tspan></text><rect data-proof-token="10" x="260" y="254" width="76" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="298" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">10</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 452" role="img" aria-labelledby="hierarchical-sum-svg-title hierarchical-sum-svg-desc" style="width:100%;max-width:32rem;display:block;margin:auto">
		<title id="hierarchical-sum-svg-title">Four ranks receive the sum 10 through a hierarchical all-reduce</title>
		<desc id="hierarchical-sum-svg-desc">Fast group A sums rank zero's 1 and rank one's 3 to get 4. Fast group B sums rank two's 2 and rank three's 4 to get 6. Only the two partials enter the slower inter-group stage, producing 10. Local distribution returns 10 to each of the original four ranks. This is a logical example, not a physical packet schedule.</desc>
		<g style="font-family:var(--font-ui);font-size:14px;fill:var(--c-text);text-anchor:middle">
			<text x="180" y="21" style="font-weight:700">1. SUM INSIDE EACH FAST GROUP</text>
			<rect x="12" y="39" width="156" height="119" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect>
			<rect x="192" y="39" width="156" height="119" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect>
			<text x="90" y="62" style="font-weight:700">GROUP A</text>
			<text x="270" y="62" style="font-weight:700">GROUP B</text>
			<text x="90" y="89">rank 0: 1</text><text x="90" y="111">rank 1: 3</text>
			<text x="270" y="89">rank 2: 2</text><text x="270" y="111">rank 3: 4</text>
			<path d="M30 122H150M210 122H330" style="stroke:var(--viz-input-stroke)"></path>
			<text x="90" y="145" style="font-weight:700">partial: 4</text><text x="270" y="145" style="font-weight:700">partial: 6</text>
			<path d="M90 158V201L133 223M270 158V201L227 223" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<rect x="12" y="184" width="336" height="122" rx="4" style="fill:none;stroke:var(--viz-edge);stroke-dasharray:5 4"></rect>
			<rect x="63" y="174" width="234" height="24" style="fill:var(--c-bg)"></rect>
			<text x="180" y="191" style="font-weight:700">2. SLOWER NETWORK</text>
			<rect x="100" y="223" width="160" height="59" rx="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<text x="180" y="248" style="font-size:20px;font-weight:700">4 + 6 = 10</text>
			<text x="180" y="269" style="font-size:13px">two group partials</text>
			<path d="M180 282V322M180 322H90V368M180 322H270V368" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path>
			<rect x="45" y="331" width="270" height="24" style="fill:var(--c-bg)"></rect>
			<text x="180" y="348" style="font-weight:700">3. COPY THE COMPLETED SUM</text>
			<rect x="12" y="368" width="156" height="72" rx="4" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect>
			<rect x="192" y="368" width="156" height="72" rx="4" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect>
			<text x="90" y="397">rank 0: <tspan style="font-weight:700">10</tspan></text><text x="90" y="422">rank 1: <tspan style="font-weight:700">10</tspan></text>
			<text x="270" y="397">rank 2: <tspan style="font-weight:700">10</tspan></text><text x="270" y="422">rank 3: <tspan style="font-weight:700">10</tspan></text>
		</g>
	</svg>
	<div class="visual-derivation-caption">Group A contributes 4 and group B contributes 6 across the slower boundary. All four original ranks receive their sum, 10. Local distribution copies the completed sum, not the original inputs. This logical example does not specify a packet schedule or establish a measured traffic saving.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="hierarchical-all-reduce-path"><strong>Check:</strong> What crosses the slow network? What identical result must every rank receive?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="hierarchical-all-reduce-path">The local sums are 1+3=4 and 2+4=6. Their global sum is 10, which every rank receives.</span></figcaption>
</figure>

This avoids sending every local copy through the slower network.

The same idea applies to any network with fast local groups and slower links among groups.

## Small example

A model uses eight-way tensor parallelism and 128-way data parallelism.

Place the tensor-parallel group inside the fastest eight-device domain because it communicates every layer. Place data parallelism across those groups because its gradient communication can overlap with backpropagation.

If the global batch becomes too small, data-parallel reductions may no longer hide under compute. More devices can then reduce efficiency instead of improving step time.

## In an interview

Use this order:

1. Draw the hardware hierarchy.
2. State bandwidth and latency for each level.
3. Identify the bytes and frequency of each communication path.
4. Put frequent activation traffic on the fastest links.
5. Estimate large-message time and small-message latency.
6. Check bisection bandwidth and possible oversubscription.
7. Compare peak specifications with measured collective speed.
8. Explain which communication can overlap with compute.

## Common mistakes

- Treating all links as equal.
- Mapping logical mesh axes without checking physical placement.
- Using peak link bandwidth as achieved collective bandwidth.
- Ignoring fixed latency for small messages.
- Spreading tensor parallelism across slow links without a cost estimate.
- Assuming more accelerators always improve throughput.
- Ignoring shared network congestion.

*Related: [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/), [all-reduce and other collectives](/concepts/all-reduce-and-collectives/), and [pipeline parallelism](/concepts/pipeline-parallelism/). Further reading: [GPU topology](https://jax-ml.github.io/scaling-book/gpus) and [TPU topology](https://jax-ml.github.io/scaling-book/tpus) in the JAX Scaling Book.*