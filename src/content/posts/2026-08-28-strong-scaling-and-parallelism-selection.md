---
title: "Strong scaling, MFU, and parallelism selection"
description: "Choose data, tensor, pipeline, and sharded parallelism by checking memory, communication, topology, and scaling efficiency."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["MFU", "model FLOPs utilization", "model flops utilisation", "scaling efficiency"]
---

## Summary

Strong scaling uses more accelerators for the same training workload. It helps only while the saved compute time is larger than added communication, idle time, and scheduling overhead.

## Why AI labs care

A model can fit on a cluster and still train slowly. Good plans answer four questions:

1. What must be split to fit in memory?
2. Which communication does each split add?
3. Which hardware links carry that communication?
4. Does adding devices reduce cost or only reduce elapsed time?

The best layout is usually the simplest layout that fits and reaches the time goal.

## Measure useful throughput

For a dense transformer with $P$ parameters and $S$ training tokens, a first compute estimate is:

$$
F_{\text{model}} \approx 6PS.
$$

If $R$ accelerators each have peak rate $C$, and the run takes time $t$, model FLOPs utilization is:

$$
\text{MFU} = \frac{F_{\text{model}}}{R C t}.
$$

MFU measures useful model work against theoretical peak compute. It includes time lost to communication, bubbles, small kernels, and idle devices.

Hardware FLOPs utilization can count recomputation and other executed work. It may be higher than MFU without improving tokens per second. State which measure you use.

## Strong-scaling efficiency

Suppose a fixed workload takes $t_1$ on one accelerator and $t_R$ on $R$ accelerators.

$$
\text{speedup} = \frac{t_1}{t_R},
\qquad
\text{efficiency} = \frac{t_1}{R t_R}.
$$

Perfect scaling has efficiency 1. In practice, efficiency falls as each device gets less compute while communication and fixed overhead remain.

If a one-device baseline cannot fit, compare two valid cluster sizes. For example, going from 64 to 128 devices has ideal speedup 2. Use the measured speedup divided by 2 as the scaling efficiency for that change.

## First check: what does not fit?

### Model state does not fit

Model state includes weights, gradients, and optimizer moments.

- Shard optimizer state first when weights and gradients still fit.
- Add gradient sharding when gradients are the problem.
- Add parameter sharding when the stored weights do not fit.
- Use tensor parallelism when one reconstructed layer or operation does not fit on one device.

### Activations do not fit

Reduce the micro-batch, checkpoint activations, or split activation axes.

- Sequence parallelism shards work around attention, such as normalization and residual operations.
- Context parallelism shards the sequence used by attention.
- Pipeline parallelism puts different layers on different stages.

### The time goal is not met

If memory already fits, add replicas or parallel work only while throughput scales well. A lower per-step time does not guarantee a lower total accelerator cost.

## What each parallelism axis costs

| Method | What it splits or saves | Main communication |
| --- | --- | --- |
| Data parallel | batch examples; model state remains replicated | gradient reduction |
| FSDP or ZeRO-3 | shard parameters, gradients, and optimizer state | parameter all-gather and gradient reduce-scatter |
| Tensor parallel | work and state inside each layer | activation collectives every layer |
| Pipeline parallel | layers across stages | point-to-point activation transfers and pipeline bubbles |
| Context parallel | long sequence activations and attention work | repeated key/value or attention communication |
| Expert parallel | expert parameters and routed work | token all-to-all |

No method is free. Add a dimension only when it solves a measured fit or speed problem.

## Place the axes on hardware

Frequent activation communication needs the fastest links. Tensor parallelism is therefore commonly placed inside one fast accelerator domain.

Pipeline communication is less frequent and often moves smaller activation boundaries, so it can cross slower links more easily.

Data-parallel and FSDP traffic can cross nodes when enough backward compute hides the communication. Small local batches reduce this overlap and hurt strong scaling.

These are placement goals, not fixed rules. Large fast networks can support wider groups. Slow local links can make even within-node sharding expensive.

## A selection procedure

### 1. Estimate memory

Count model state, saved activations, temporary buffers, and safety margin. Use measured peak memory when code exists.

### 2. Use the smallest required model-parallel group

If one layer fits on one device, do not add tensor parallelism only because it is common. If the full model fits with optimizer-state sharding, avoid parameter all-gathers unless they improve another constraint.

### 3. Set micro-batch and accumulation

For data-parallel degree $D_p$, local micro-batch size $B$, sequence length $T$, and $G$ accumulated micro-batches:

$$
\text{tokens per optimizer step} = D_p B T G.
$$

Pipeline schedules may split these micro-batches further. Keep the desired global batch and optimization behavior fixed when comparing layouts.

### 4. Estimate exposed communication

Do not add all communication times blindly. Some traffic overlaps with compute. The exposed part is what extends the critical path.

Measure collective time, overlap, and idle gaps in a trace.

### 5. Test a small layout grid

Compare a few legal choices for tensor, pipeline, context, and data-parallel degrees. Record:

- peak memory;
- tokens per second;
- MFU;
- step-time variance;
- communication time;
- convergence or numerical changes.

### 6. Stop scaling when efficiency is poor

More devices can still reduce wall time while increasing accelerator-hours. Decide whether the deadline is worth the extra cost.

## Pipeline bubbles

For $P_p$ pipeline stages and $M$ micro-batches, a simple schedule has approximate bubble fraction:

$$
\frac{P_p - 1}{M + P_p - 1}.
$$

More micro-batches reduce the bubble. They can increase activation memory or change the effective batch schedule. Interleaved schedules change the exact formula.

## In an interview

Use this order:

1. Estimate parameters, state memory, activations, and compute.
2. Identify the first memory constraint.
3. Pick the smallest sharding group that resolves it.
4. Map frequent communication to the fastest links.
5. Preserve the intended global batch.
6. Estimate MFU, wall time, and accelerator-hours.
7. Name what can overlap and what remains exposed.
8. Propose a short measured layout sweep.

## Common mistakes

- Starting with three-dimensional parallelism before checking what fits.
- Comparing layouts with different global batches.
- Reporting peak FLOPs instead of MFU.
- Calling recomputation useful model work.
- Assuming all communication overlaps.
- Using more devices after strong-scaling efficiency has collapsed.
- Optimizing wall time while ignoring total accelerator-hours.

*Related: [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/), [accelerator network topology](/concepts/accelerator-network-topology/), and [pipeline parallelism](/concepts/pipeline-parallelism/). Further practice: [training parallelism in the JAX Scaling Book](https://jax-ml.github.io/scaling-book/training).*