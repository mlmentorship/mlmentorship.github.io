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
