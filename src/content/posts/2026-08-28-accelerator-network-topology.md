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