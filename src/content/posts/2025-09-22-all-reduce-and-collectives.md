---
title: "All-reduce and other collectives"
description: "Choose a collective from each rank's required output, then account for reduction, shard ownership, traffic, latency and overlap."
date: "2025-09-22"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["allreduce", "allgather", "reduce scatter", "collective communication"]
---

## Summary

Choose a collective by each rank's required output: concatenate pieces, combine aligned values, or distribute a completed result.

## Problem

The process group contains ranks 0 and 1, holding `[1, 2]` and `[3, 4]`.
Both buffers have shape `(2,)`, dtype int64 and the same index meanings.
Specify what each rank holds after a sum all-reduce, reduce-scatter and all-gather.
Use rank order for concatenation and equal, index-aligned chunks for scattering.

## Picture: Add Aligned Positions or Join Pieces

Sum reduction combines corresponding positions: `[1+3, 2+4]` becomes `[4, 6]`.
All-reduce gives that result to both ranks.
Reduce-scatter keeps `[4]` on rank 0 and `[6]` on rank 1.
All-gather of those reduced shards restores `[4, 6]` on both.

All-gather of the original buffers instead produces `[1, 2, 3, 4]`.
It concatenates; it does not add.
The all-reduce identity therefore needs compatible reduction, chunk boundaries and rank order.

<span id="the-five-primitives"></span>

## Six Output Contracts

<span id="broadcast"></span>
<span id="reduce"></span>
<span id="all-reduce"></span>
<span id="all-gather"></span>
<span id="reduce-scatter"></span>

| Operation on the stated inputs | Rank 0 afterward | Rank 1 afterward |
| --- | --- | --- |
| Broadcast rank 0 | `[1, 2]` | `[1, 2]` |
| Sum reduce to rank 0 | `[4, 6]` | No required result |
| Sum all-reduce | `[4, 6]` | `[4, 6]` |
| All-gather original buffers | `[1, 2, 3, 4]` | `[1, 2, 3, 4]` |
| Sum reduce-scatter | `[4]` | `[6]` |
| All-to-all, one value per destination | `[1, 3]` | `[2, 4]` |

Sum does not divide by rank count.
Equal-sized data-parallel batches can average by dividing the summed gradients appropriately.
Unequal local batches require weighting by example counts, not blindly averaging local means.

## Reference the Values Before the Network

This reference models values and order, not transport, dtype enforcement or synchronization.

```python
def sum_by_index(buffers):
	if not buffers:
		raise ValueError("the process group must be nonempty")
	width = len(buffers[0])
	if any(len(buffer) != width for buffer in buffers):
		raise ValueError("rank inputs must have matching shapes")
	return [sum(contributions) for contributions in zip(*buffers)]


def reduce_scatter_reference(buffers):
	reduced = sum_by_index(buffers)
	ranks = len(buffers)
	if len(reduced) % ranks:
		raise ValueError("equal chunks require a divisible width")
	width = len(reduced) // ranks
	return [reduced[rank * width:(rank + 1) * width] for rank in range(ranks)]


def all_gather_reference(shards):
	if not shards or any(len(shard) != len(shards[0]) for shard in shards):
		raise ValueError("this reference requires equal-sized rank shards")
	return [value for shard in shards for value in shard]


buffers = [[1, 2], [3, 4]]
assert sum_by_index(buffers) == [4, 6]
assert reduce_scatter_reference(buffers) == [[4], [6]]
assert all_gather_reference(reduce_scatter_reference(buffers)) == [4, 6]
assert all_gather_reference(buffers) == [1, 2, 3, 4]
```

## The Ring All-Reduce

Now use four ranks and full input size $B=8$ MB, with decimal MB.
Split each input into four equal, aligned chunks.
For chunk A, rank-local contributions are $a_0,a_1,a_2,a_3$.
Three receive-and-add steps finish $A^*$; three forwarding steps distribute it without further addition.
Other chunks follow the corresponding schedule concurrently.

Each rank sends $B/4=2$ MB per step: six steps send 12 MB.
It also receives 12 MB; do not count sent-plus-received bytes as sent traffic.
Generally, ring sent traffic is $2B(N-1)/N$ per rank over $2(N-1)$ steps.

<!-- visual:ring-all-reduce-chunk-trace -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="ring-trace-title">
	<p class="visual-kicker">One chunk, four ranks</p>
	<p class="visual-title" id="ring-trace-title">Reduction moves the partial sum; gathering moves the finished chunk.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 370" style="min-width: 800px" role="img" aria-labelledby="ring-trace-svg-title ring-trace-svg-desc">
			<title id="ring-trace-svg-title">One chunk traced through a four-rank ring all-reduce</title>
			<desc id="ring-trace-svg-desc">In reduce-scatter, rank zero sends its contribution a zero to rank one, which adds a one. Rank one sends that partial sum to rank two, which adds a two. Rank two sends the next partial sum to rank three, which adds a three and finishes chunk A. In all-gather, rank three sends the finished chunk to rank zero, then rank zero forwards it to rank one, and rank one forwards it to rank two without further addition. Every rank concurrently sends one chunk of size B divided by four in each of the six steps, for three B divided by two bytes per rank.</desc>
			<text class="viz-axis-label" x="20" y="24">REDUCE-SCATTER · 3 hops · receive, then add the local contribution</text>
			<g aria-label="Reduce-scatter trace">
				<rect class="viz-node viz-node--input" x="20" y="48" width="142" height="62" rx="4"></rect>
				<text class="viz-node-value" x="91" y="70">START AT RANK 0</text>
				<text class="viz-node-label" x="91" y="94">a₀</text>
				<path class="viz-axis" d="M162 79 H210"></path><path class="viz-arrow-forward" d="M210 79 l-9 -5 v10 Z"></path>
				<text class="viz-edge-label" x="186" y="67">step 1</text>
				<rect class="viz-node viz-node--focus" x="210" y="48" width="142" height="62" rx="4"></rect>
				<text class="viz-node-value" x="281" y="70">AT RANK 1</text>
				<text class="viz-node-label" x="281" y="94">a₀ + a₁</text>
				<path class="viz-axis" d="M352 79 H400"></path><path class="viz-arrow-forward" d="M400 79 l-9 -5 v10 Z"></path>
				<text class="viz-edge-label" x="376" y="67">step 2</text>
				<rect class="viz-node viz-node--focus" x="400" y="48" width="142" height="62" rx="4"></rect>
				<text class="viz-node-value" x="471" y="70">AT RANK 2</text>
				<text class="viz-node-label" x="471" y="94">a₀ + a₁ + a₂</text>
				<path class="viz-axis" d="M542 79 H590"></path><path class="viz-arrow-forward" d="M590 79 l-9 -5 v10 Z"></path>
				<text class="viz-edge-label" x="566" y="67">step 3</text>
				<rect class="viz-node viz-node--output" x="590" y="48" width="150" height="62" rx="4"></rect>
				<text class="viz-node-value" x="665" y="70">FINAL A* AT RANK 3</text>
				<text class="viz-node-label" x="665" y="94">a₀ + a₁ + a₂ + a₃</text>
			</g>
			<text class="viz-label" x="20" y="135">All four chunks follow the same schedule concurrently; A is isolated here so its accumulation stays visible.</text>
			<path class="viz-baseline" d="M20 153 H740"></path>
			<text class="viz-axis-label" x="20" y="181">ALL-GATHER · 3 hops · forward the finished A* without adding</text>
			<g aria-label="All-gather trace">
				<rect class="viz-node viz-node--output" x="20" y="205" width="142" height="58" rx="4"></rect>
				<text class="viz-node-value" x="91" y="227">RANK 3 SENDS</text>
				<text class="viz-node-label" x="91" y="249">A*</text>
				<path class="viz-baseline" d="M162 234 H210"></path><path class="viz-arrow-forward" d="M210 234 l-9 -5 v10 Z"></path>
				<text class="viz-edge-label" x="186" y="222">step 1</text>
				<rect class="viz-node" x="210" y="205" width="142" height="58" rx="4"></rect>
				<text class="viz-node-value" x="281" y="227">RANK 0 RECEIVES</text>
				<text class="viz-node-label" x="281" y="249">A*</text>
				<path class="viz-baseline" d="M352 234 H400"></path><path class="viz-arrow-forward" d="M400 234 l-9 -5 v10 Z"></path>
				<text class="viz-edge-label" x="376" y="222">step 2</text>
				<rect class="viz-node" x="400" y="205" width="142" height="58" rx="4"></rect>
				<text class="viz-node-value" x="471" y="227">RANK 1 RECEIVES</text>
				<text class="viz-node-label" x="471" y="249">A*</text>
				<path class="viz-baseline" d="M542 234 H590"></path><path class="viz-arrow-forward" d="M590 234 l-9 -5 v10 Z"></path>
				<text class="viz-edge-label" x="566" y="222">step 3</text>
				<rect class="viz-node" x="590" y="205" width="150" height="58" rx="4"></rect>
				<text class="viz-node-value" x="665" y="227">RANK 2 RECEIVES</text>
				<text class="viz-node-label" x="665" y="249">A*</text>
			</g>
			<text class="viz-axis-label" x="20" y="301">PER-RANK SENT BYTES FOR N = 4</text>
			<rect class="viz-node viz-node--focus" x="20" y="315" width="720" height="38" rx="4"></rect>
			<text class="viz-callout" x="380" y="339" text-anchor="middle">[3 reduce-scatter sends + 3 all-gather sends] × B/4 = 6B/4 = 3B/2</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> three receive-and-add steps build A*. Three forwarding steps distribute that completed chunk. All chunks follow the schedule concurrently. Each rank sends six chunks of size B/4, totaling 3B/2 bytes. This is sent traffic, not sent-plus-received traffic. Sources: <a href="https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/">Baidu ring explanation</a> and <a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html">NCCL semantics</a>.</figcaption>
</figure>

## Change 1: The Consumer Needs Only Its Shard

**New task.** The optimizer needs its own portion of the summed gradient, not the full sum everywhere.

**Picture and change.** Stop after reduce-scatter and preserve the optimizer's shard mapping.
Do not all-gather a result merely to discard the other shards.

**Check.** The two-rank outputs remain `[4]` and `[6]`, not two full buffers.
In the four-rank ring model, this phase sends 6 MB per rank instead of 12 MB.

**Use when.** Downstream work can consume sharded results.

## Change 2: Thousands of Tiny Buffers

**New task.** Replace one 8 MB buffer with 1,000 separate 8 KB buffers.

**Picture and change.** Total ring bytes stay fixed, but sequential communication steps multiply.
For effective bandwidth $b$ and per-step latency $\alpha$, use the simplified unoverlapped model

$$
T_{\mathrm{ring}}\approx2(N-1)\alpha+\frac{2B(N-1)}{Nb}.
$$

At four ranks, $\alpha=5$ microseconds and $b=1$ GB/s give 12.03 ms for the single buffer.
The 1,000 serial collectives take 42 ms in this model, despite sending the same 12 MB per rank.

**Check.** Measure backend launch/hop costs, bucketing, tree alternatives and actual topology before selecting an algorithm.
These calculations are not hardware measurements.

**Use when.** Fixed overhead dominates small messages.

## Failure: Ranks Disagree About the Operation

A skipped call or mismatched shape, dtype, chunk order or collective sequence can hang, fail or invalidate results.
All participating ranks must follow the process group's matching communication contract.

## Transfer Check

Does a long total collective duration establish the training bottleneck?

### Reasoned Answer

No. Measure its exposed critical-path tail after overlap and contention.
Summed durations can include work already hidden behind computation.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Where Each Appears

| Distributed pattern | Collectives |
|---------------------|------------|
| **DDP** (data parallel) | Gradient all-reduce at synchronized backward boundaries; accumulation can defer synchronization |
| **FSDP / ZeRO-3** | Parameter all-gather when needed; gradient reduce-scatter under the selected sharding strategy |
| **Tensor parallelism** | All-reduce, reduce-scatter or all-gather according to input/output layouts |
| **Pipeline parallelism** | Point-to-point sends (not collective) between adjacent stages |
| **Expert parallelism (MoE)** | All-to-all to route tokens to experts |
| **Embedding lookup at scale** | All-to-all to gather sharded embedding rows |

## All-to-all

Each rank sends destination-specific pieces; receive order follows the agreed source layout.
MoE routing and layout changes use this exchange.
Cost depends on peer traffic, balance, bisection bandwidth and topology, not the operation name alone.

## Hardware backends

- **NCCL:** NVIDIA accelerator communication.
- **RCCL:** AMD accelerator communication.
- **MPI:** broader high-performance message passing.
- **Gloo:** commonly used for CPU collectives and supported distributed interfaces.

Supported operations and selected algorithms depend on version, hardware, group and message size.

## Bandwidth and topology

Distinguish local accelerator links from inter-node or inter-slice networks.
Hierarchical algorithms combine local and scale-out phases.
Use measured effective bandwidth for the message and schedule, not advertised peak bandwidth.

## Cost model in DDP

For $P$ parameters and gradient element size $s$, $B=Ps$ bytes in the ring model.
Suppose seven billion gradients use BF16, 32 ranks participate, and effective ring bandwidth is 200 GB/s.
Then $B=14$ GB and per-rank sent traffic is 27.125 GB.
The bandwidth term is 135.625 ms, not a measured step time.

## Common pitfalls

Buckets trade fixed overhead against readiness, memory and overlap.
DDP/FSDP overlap depends on configuration and schedule; inspect traces rather than assuming it.
Reduction order can change floating-point rounding even when mathematical outputs agree.
The integer reference does not model those numerical differences or diagnose distributed hangs.

## Related

- [Tensor parallelism](/concepts/tensor-parallelism/). Heavy collective user.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Uses all-gather and reduce-scatter.
- [Pipeline parallelism](/concepts/pipeline-parallelism/). Uses point-to-point, not collectives.
- [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/). Select collectives from array layouts.
- [Accelerator network topology](/concepts/accelerator-network-topology/). Understand the links that carry them.

</div>
