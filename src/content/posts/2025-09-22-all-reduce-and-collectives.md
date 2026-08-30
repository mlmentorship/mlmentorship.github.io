---
title: "All-reduce and other collectives"
description: "The communication primitives behind every distributed training job. All-reduce, all-gather, reduce-scatter, broadcast. What they do, costs, and when each is used."
date: "2025-09-22"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["allreduce", "allgather", "reduce scatter", "collective communication"]
---

## Summary

A **collective communication operation** is a coordinated message-passing primitive across a group of processes. The five most common in deep learning: **broadcast** (one-to-all copy), **reduce** (all-to-one sum), **all-reduce** (all-to-all sum into all), **all-gather** (concatenate everyone's data into all), **reduce-scatter** (sum-then-shard).

Every distributed training run is built on these primitives. Knowing which collective is invoked when is the difference between explaining "DDP all-reduces gradients" and actually understanding the cost of FSDP, TP, or pipeline parallelism. Communication time is often the bottleneck. Collective choice determines throughput.

## The five primitives

Let $N$ be the number of ranks and $B$ be the size of the full logical tensor. A useful communication model is:

$$
T \approx \alpha \times \text{communication steps} + \frac{\text{bytes moved per rank}}{\text{effective bandwidth}},
$$

where $\alpha$ is fixed latency per step. The exact algorithm depends on message size and topology.

### Broadcast
One process sends its buffer to all others.
- **Traffic**: each receiver obtains $B$ bytes. A tree lowers the number of sequential hops for small messages.
- **Use**: distribute initial weights, broadcast a hyperparameter.

### Reduce
All processes contribute; one receives the sum (or max, min, etc.).
- **Traffic**: the root receives a reduced tensor of size $B$. Ring and tree algorithms divide the work differently.
- **Use**: aggregating metrics to rank 0 for logging.

### All-reduce
All processes contribute and all receive the sum.
- **Ring traffic per rank**: $2B(N-1)/N$, split across reduce-scatter and all-gather phases.
- **Use**: gradient aggregation in DDP; output sum in tensor parallelism.

### All-gather
Each process has a chunk of size $B/N$; all end with the full $B$ concatenation.
- **Ring traffic per rank**: $B(N-1)/N$.
- **Use**: FSDP. Gather sharded parameters before forward pass.

### Reduce-scatter
Each process contributes a buffer of size $B$; all end with their slice $B/N$ of the sum.
- **Ring traffic per rank**: $B(N-1)/N$.
- **Use**: FSDP. Sum gradients and keep only your shard.

**Identity**: all-reduce $=$ reduce-scatter $+$ all-gather.

## The ring all-reduce

The dominant implementation in 2026 (Baidu Ring, Horovod, NCCL):

1. Each process splits its buffer into $N$ chunks.
2. **Reduce-scatter phase** ($N - 1$ steps): each process sends one chunk to its right neighbor, receives one from the left, accumulates.
3. **All-gather phase** ($N - 1$ steps): each process has the final value of one chunk; cycle around so everyone has the full buffer.

Total: $2(N-1)$ steps, each transferring $B/N$ bytes per rank. **Traffic per rank is $2B(N-1)/N$, which approaches $2B$.** A ring uses link bandwidth well for large messages. Its many sequential steps can make small messages latency-bound.

<!-- visual:ring-all-reduce-chunk-trace -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="ring-trace-title">
	<p class="visual-kicker">One chunk, four ranks</p>
	<p class="visual-title" id="ring-trace-title">Reduction moves the partial sum; gathering moves the finished chunk.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 370" role="img" aria-labelledby="ring-trace-svg-title ring-trace-svg-desc">
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
			<text class="viz-axis-label" x="20" y="301">PER-RANK TRAFFIC FOR N = 4</text>
			<rect class="viz-node viz-node--focus" x="20" y="315" width="720" height="38" rx="4"></rect>
			<text class="viz-callout" x="380" y="339" text-anchor="middle">[3 reduce-scatter sends + 3 all-gather sends] × B/4 = 6B/4 = 3B/2</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow chunk A from left to right. Three reduce-scatter hops build A* from four rank-local contributions; three all-gather hops copy A* to the other ranks without more arithmetic. Every rank does this concurrently for one B/4 chunk per step, so it sends 6 × B/4 = 3B/2 bytes, exactly 2B(N − 1)/N for N = 4. Original schematic based on the <a href="https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/">Baidu ring all-reduce explanation</a> and <a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html">NCCL collective semantics</a>.</figcaption>
</figure>

## Where each appears

| Distributed pattern | Collectives |
|---------------------|------------|
| **DDP** (data parallel) | All-reduce on gradients per backward pass |
| **FSDP / ZeRO-3** | All-gather on parameters before forward; reduce-scatter on gradients after backward |
| **Tensor parallelism** | All-reduce on activations after each parallel matmul |
| **Pipeline parallelism** | Point-to-point sends (not collective) between adjacent stages |
| **Expert parallelism (MoE)** | All-to-all to route tokens to experts |
| **Embedding lookup at scale** | All-to-all to gather sharded embedding rows |

## All-to-all

A sixth primitive: each process sends a different chunk to every other process. It is used in MoE token routing and to move a tensor split from one axis to another. Its cost depends on bytes per peer, network bisection bandwidth, routing balance, and topology. It is often a bottleneck, but it is not always more expensive than every other collective.

## Hardware backends

- **NCCL** (Nvidia): the dominant backend on Nvidia GPUs; ring + tree implementations, NVLink and InfiniBand aware.
- **RCCL**: AMD equivalent.
- **MPI**: classical HPC backend; used outside ML.
- **Gloo**: PyTorch CPU collective backend (slow).

## Bandwidth and topology

At least two physical bandwidth levels matter:

- **Inside a fast accelerator domain**: links such as NVLink, NVSwitch, or an accelerator torus.
- **Across nodes or slices**: networks such as InfiniBand or Ethernet with RDMA.

A large cluster usually has faster local groups and slower links between groups. Hierarchical collectives reduce or gather inside each local group before using the scale-out network. Use measured bandwidth for the target message size rather than a peak link specification.

## Cost model in DDP

For a model with $P$ parameters, gradient dtype size $s$, and $N$ data-parallel ranks, a ring gradient all-reduce has:

- Logical gradient tensor: $B = Ps$ bytes.
- Traffic per rank: $2Ps(N-1)/N$ bytes.
- Large-message lower bound: traffic divided by effective ring bandwidth.

For a 7B parameter model with BF16 gradients, $B=14$ GB. On 32 ranks at 200 GB/s effective ring bandwidth, the bandwidth lower bound is about 136 ms. Fixed latency and contention can increase it. **Gradient bucketing** avoids tiny messages. **Overlap with backward** can hide part of the reduction, so the exposed time in a step trace is more important than total collective duration.

## Common pitfalls

- **All-reducing every parameter separately.** Tiny messages are latency-dominated. Use gradient buckets and tune their size, such as PyTorch's `bucket_cap_mb` setting.
- **No overlap with compute.** PyTorch DDP overlaps automatically; FSDP needs explicit configuration (`forward_prefetch`, `backward_prefetch`).
- **Mixed dtypes across ranks.** All-reduce requires identical dtype on all ranks; mismatch → cryptic NCCL error.
- **Hangs from rank desync.** If one rank skips a collective (e.g., divergent code path), all others hang waiting. Use the same control flow on every rank.
- **Ranking collectives by name alone.** Cost follows bytes, message size, algorithm, balance, and topology. All-to-all is difficult on oversubscribed networks, while a very large all-reduce can still cost more.

## Related

- [Tensor parallelism](/concepts/tensor-parallelism/). Heavy collective user.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Uses all-gather and reduce-scatter.
- [Pipeline parallelism](/concepts/pipeline-parallelism/). Uses point-to-point, not collectives.
- [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/). Select collectives from array layouts.
- [Accelerator network topology](/concepts/accelerator-network-topology/). Understand the links that carry them.
