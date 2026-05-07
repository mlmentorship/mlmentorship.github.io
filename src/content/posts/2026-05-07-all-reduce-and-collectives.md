---
title: "All-reduce and other collectives"
description: "The communication primitives behind every distributed training job. All-reduce, all-gather, reduce-scatter, broadcast. What they do, costs, and when each is used."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A **collective communication operation** is a coordinated message-passing primitive across a group of processes. The five most common in deep learning: **broadcast** (one-to-all copy), **reduce** (all-to-one sum), **all-reduce** (all-to-all sum into all), **all-gather** (concatenate everyone's data into all), **reduce-scatter** (sum-then-shard).

## Why it matters

Every distributed training run is built on these primitives. Knowing which collective is invoked when is the difference between explaining "DDP all-reduces gradients" and actually understanding the cost of FSDP, TP, or pipeline parallelism. Communication time is often the bottleneck. Collective choice determines throughput.

## The five primitives

For $N$ processes each with a buffer of size $B$:

### Broadcast
One process sends its buffer to all others.
- **Cost**: $O(B)$ (with tree-based implementation: $O(B \log N)$ per step, $\log N$ steps).
- **Use**: distribute initial weights, broadcast a hyperparameter.

### Reduce
All processes contribute; one receives the sum (or max, min, etc.).
- **Cost**: $O(B \log N)$.
- **Use**: aggregating metrics to rank 0 for logging.

### All-reduce
All processes contribute and all receive the sum.
- **Cost**: $O(B)$ with ring algorithm; $O(B \log N)$ with tree.
- **Use**: gradient aggregation in DDP; output sum in tensor parallelism.

### All-gather
Each process has a chunk of size $B/N$; all end with the full $B$ concatenation.
- **Cost**: $O(B)$ with ring.
- **Use**: FSDP. Gather sharded parameters before forward pass.

### Reduce-scatter
Each process contributes a buffer of size $B$; all end with their slice $B/N$ of the sum.
- **Cost**: $O(B)$ with ring.
- **Use**: FSDP. Sum gradients and keep only your shard.

**Identity**: all-reduce $=$ reduce-scatter $+$ all-gather.

## The ring all-reduce

The dominant implementation in 2026 (Baidu Ring, Horovod, NCCL):

1. Each process splits its buffer into $N$ chunks.
2. **Reduce-scatter phase** ($N - 1$ steps): each process sends one chunk to its right neighbor, receives one from the left, accumulates.
3. **All-gather phase** ($N - 1$ steps): each process has the final value of one chunk; cycle around so everyone has the full buffer.

Total: $2(N-1)$ steps, each transferring $B/N$ bytes per process. **Aggregate bandwidth: $2B(N-1)/N \to 2B$**, independent of $N$. This is why ring all-reduce scales gracefully.

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

A sixth primitive: each process sends a different chunk to every other process. Used in MoE for token routing across expert-parallel ranks. Bandwidth-intensive; often the dominant cost in MoE training and serving.

## Hardware backends

- **NCCL** (Nvidia): the dominant backend on Nvidia GPUs; ring + tree implementations, NVLink and InfiniBand aware.
- **RCCL**: AMD equivalent.
- **MPI**: classical HPC backend; used outside ML.
- **Gloo**: PyTorch CPU collective backend (slow).

## Bandwidth and topology

Two physical bandwidths matter:

- **Intra-node (NVLink, NVSwitch, Infinity Fabric)**: ~600 GB/s for NVLink 4 (H100), ~900 GB/s for NVLink 5 (B100/B200).
- **Inter-node (InfiniBand, Ethernet RDMA)**: 200–800 Gbps per link.

A typical large training cluster has a hierarchy: NVLink within node, IB across nodes. Collectives are designed to use intra-node bandwidth first, then aggregate across nodes. Hierarchical NCCL.

## Cost model in DDP

For a model with $P$ parameters and $N$ GPUs, gradient all-reduce per step:

- Bytes per GPU: $P \cdot \text{dtype\_size}$.
- Wall-clock: $\sim 2P \cdot \text{dtype\_size} / \text{bandwidth}$.

For a 7B parameter model in BF16 (14 GB per copy) on 32 GPUs with 200 GB/s effective bandwidth: ~140 ms per all-reduce. Becomes the bottleneck at small batch sizes. **Gradient bucketing** (batching multiple parameters into one all-reduce call) and **overlap with backward** (start all-reducing earlier layers while later layers are still computing) hide most of this.

## Common pitfalls

- **All-reducing every parameter separately.** Tiny messages; latency-dominated. Use bucketing (PyTorch's `find_unused_parameters` and `bucket_cap_mb` tune this).
- **No overlap with compute.** PyTorch DDP overlaps automatically; FSDP needs explicit configuration (`forward_prefetch`, `backward_prefetch`).
- **Mixed dtypes across ranks.** All-reduce requires identical dtype on all ranks; mismatch → cryptic NCCL error.
- **Hangs from rank desync.** If one rank skips a collective (e.g., divergent code path), all others hang waiting. Use the same control flow on every rank.
- **Treating all-to-all as cheap.** It's the most expensive collective; MoE communication often dominates training cost.

## Related

- [Tensor parallelism](/reference/tensor-parallelism/). Heavy collective user.
- [FSDP and ZeRO](/reference/fsdp-and-zero/). Uses all-gather and reduce-scatter.
- [Pipeline parallelism](/reference/pipeline-parallelism/). Uses point-to-point, not collectives.
