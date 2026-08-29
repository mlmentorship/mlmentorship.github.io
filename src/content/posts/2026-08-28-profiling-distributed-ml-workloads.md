---
title: "Profiling distributed ML workloads"
description: "Use step traces, roofline limits, and communication timelines to find the exposed bottleneck in training and inference."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Distributed ML profiling compares a measured step with compute, memory, and network limits. It finds which work extends the critical path and tests one repair at a time.

## Why AI labs care

Low utilization is a symptom, not a diagnosis. It can come from:

- input stalls;
- many small kernels;
- memory-bound operations;
- exposed collectives;
- pipeline bubbles;
- device imbalance;
- host synchronization;
- compilation or memory allocation;
- failed overlap between communication and compute.

A trace makes these causes visible.

## Start with stable measurements

Before reading one trace:

1. verify correctness;
2. exclude compilation and warm-up steps;
3. use a representative batch and sequence-length mix;
4. record several steady-state steps;
5. report the median and tail step time;
6. save configuration, software versions, and topology.

Do not compare traces from different batch sizes without saying so.

## Build three lower bounds

### Compute bound

$$
T_{\text{compute}} \geq \frac{\text{useful model FLOPs}}{\text{peak FLOPs}}.
$$

Measured model FLOPs utilization compares this lower bound with wall time.

### Memory bound

For each important kernel:

$$
T_{\text{memory}} \geq \frac{\text{bytes moved}}{\text{effective memory bandwidth}}.
$$

The roofline model compares arithmetic intensity with the machine's compute-to-bandwidth ratio.

### Communication bound

$$
T_{\text{comm}} \geq T_{\text{fixed}} + \frac{\text{bytes moved per device}}{\text{effective collective bandwidth}}.
$$

Use measured collective bandwidth for the same message size and device group. Peak link bandwidth is not enough.

These bounds can overlap. Do not add all three as if every event were serial.

## Read the trace from the outside in

### 1. Mark the training or inference step

Find true boundaries and synchronize only when measurement requires it. Hidden asynchronous work can make one operation appear cheap while a later wait pays its cost.

### 2. Find large idle gaps

Ask what each accelerator is waiting for:

- input data;
- another rank;
- a collective;
- a host callback;
- a pipeline stage;
- memory allocation or compilation.

### 3. Find the longest exposed operations

An operation can consume much total time without extending the step if it overlaps with other work. Focus on the critical path.

### 4. Group kernels by type

Classify matrix multiplications, attention, elementwise operations, memory copies, and collectives. Many tiny elementwise kernels may point to missing fusion. Poorly shaped matrix multiplications may fail to use tensor units well.

### 5. Compare ranks

One slow rank can hold every other rank at a collective. Compare start times, kernel durations, input lengths, network paths, and device clocks.

## Common trace patterns

| Trace pattern | Likely cause | First check |
| --- | --- | --- |
| Device idle before each step | input pipeline | batch-ready time and host work |
| Many tiny kernels with gaps | launch overhead or missing fusion | operation fusion and shapes |
| Long matrix operations with high unit use | compute-bound work | model FLOPs and kernel shape |
| Long memory stalls with low arithmetic intensity | memory-bound work | bytes moved and data reuse |
| Collective starts after backward compute ends | no overlap | gradient bucket timing |
| Collective overlaps most compute, then has a long tail | last bucket exposed | bucket order, size, and straggler |
| Regular empty pipeline regions | pipeline bubble | stage count, micro-batches, balance |
| One rank arrives late at every collective | load or input imbalance | per-rank work and network placement |
| Repeated reshard operations | poor layout choices | producer and consumer tensor layouts |

For one late rank, compare input-ready time, backward compute duration, and accelerator utilization across ranks. A late input points to the data path. A longer backward pass points to uneven shapes, extra work, throttling, or a slow device. Similar compute followed by a longer collective points to network placement or contention.

## Profiling a sharded program

For every large tensor, record its layout before and after an operation. A compiler or framework may insert:

- all-gather;
- reduce-scatter;
- all-reduce;
- all-to-all;
- host-device transfer.

Unexpected resharding often comes from incompatible layouts between adjacent operations. Removing one layout change can matter more than tuning a kernel.

## Profiling compilation

A framework may show high-level operations while the accelerator executes a compiled graph.

Inspect both levels:

- source-level operation and tensor shape;
- compiled fusion or kernel;
- device timeline;
- communication events;
- memory allocation and peak use.

Do not assume a source operation maps to one kernel.

## Run controlled experiments

Keep an experiment table:

| Field | Record |
| --- | --- |
| Hypothesis | what limits the step |
| Change | one controlled modification |
| Prediction | expected direction and size |
| Correctness | invariant or test result |
| Median result | stable step time or throughput |
| Trace evidence | what moved on the critical path |
| Decision | keep, revert, or test again |

Useful controlled changes include message size, bucket size, local batch, sequence length, sharding layout, data-loader workers, and kernel choice.

## In an interview

Use this order:

1. Define workload, metric, and correctness test.
2. Remove warm-up and collect several steady steps.
3. Compute simple compute, memory, and communication bounds.
4. Find idle gaps and the critical path.
5. Compare devices or ranks.
6. Check for unexpected layout changes and exposed collectives.
7. State one hypothesis with a predicted result.
8. Change one mechanism, measure again, and keep a ledger.

## Common mistakes

- Optimizing from average utilization alone.
- Treating total operation time as exposed time.
- Using peak bandwidth instead of measured bandwidth.
- Measuring only one step.
- Including compilation in steady-state throughput.
- Changing batch size and calling the result a kernel speedup.
- Looking at one rank only.
- Making several changes before collecting a new trace.
- Accepting a speedup without checking model outputs.

*Related: [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/), [strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/), and [optimize an accelerator workload](/questions/optimize-accelerator-workload/). Further reading: [profiling in the JAX Scaling Book](https://jax-ml.github.io/scaling-book/profiling).*