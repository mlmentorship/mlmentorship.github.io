---
title: "Plan and cost a 70B transformer training run"
description: "Turn a model configuration and cluster budget into parameters, memory, parallelism, time, and an experiment plan."
date: "2026-08-28"
draft: false
tags: ["questions"]
category: "questions"
---

> You need to train a dense decoder-only transformer on 300 billion tokens. It has 80 layers, width 8,192, gated feed-forward width 28,672, 64 query heads, 8 key/value heads, head size 128, and a tied vocabulary of 128,000 tokens. You have 512 GPUs in 64 eight-GPU nodes. Each GPU has 80 GB of memory, about 1 PFLOP/s peak BF16 compute, a fast switched link inside its node, and 45 GB/s effective per-rank transfer bandwidth for large cross-node collectives under load. Plan the run.

This question tests whether a candidate can turn dimensions into a measured systems plan. There is no single correct parallel layout. Assumptions and checks matter more than naming a framework.

## Try it before reading the answer

Estimate:

1. total parameters;
2. model-state memory;
3. total training FLOPs;
4. wall time at 50% model FLOPs utilization;
5. accelerator-hours;
6. a first parallel layout;
7. global batch construction for about four million tokens per optimizer step;
8. the first experiments and failure checks.

Assume BF16 weights and gradients, FP32 Adam moments, and no separate FP32 master weights.

## Count the parameters

For each layer, gated feed-forward parameters are:

$$
3DF = 3(8192)(28672) \approx 704.6\text{ million}.
$$

Attention parameters with grouped-query attention are:

$$
2D^2 + 2DKH
= 2(8192)^2 + 2(8192)(8)(128)
\approx 151.0\text{ million}.
$$

Across 80 layers:

$$
80(704.6 + 151.0)\text{ million} \approx 68.4\text{ billion}.
$$

The tied token embedding adds:

$$
VD = 128000(8192) \approx 1.05\text{ billion}.
$$

Norms and small terms bring the estimate close to 70 billion parameters.

## Count the model state

Per parameter:

- BF16 weight: 2 bytes;
- BF16 gradient: 2 bytes;
- two FP32 Adam moments: 8 bytes.

Total model state is about:

$$
70\text{B} \times 12\text{ bytes} = 840\text{ GB}.
$$

This excludes activations, temporary kernel buffers, communication buffers, and allocator margin.

## Count the compute

Start with the parameter-matrix estimate:

$$
6 \times 70\text{B parameters} \times 300\text{B tokens}
= 1.26 \times 10^{23}\text{ FLOPs}.
$$

This estimate should be refined for long-context attention and the final architecture. It is sufficient for an early capacity plan.

At sequence length 8,192, the attention score and weighted-value operations are large enough to include. Their forward and backward cost is approximately:

$$
12 S T D L
= 12(300\text{B})(8192)(8192)(80)
\approx 1.93 \times 10^{22}\text{ FLOPs}.
$$

The refined total is about $1.45 \times 10^{23}$ FLOPs, before smaller operations.

The cluster peak is:

$$
512 \times 10^{15} = 5.12 \times 10^{17}\text{ FLOP/s}.
$$

At 50% MFU, useful model throughput is $2.56 \times 10^{17}$ FLOP/s. Using the refined compute estimate, the run time is:

$$
\frac{1.45 \times 10^{23}}{2.56 \times 10^{17}}
\approx 568{,}000\text{ seconds}
\approx 6.6\text{ days}.
$$

The run uses about:

$$
6.6 \times 24 \times 512 \approx 81{,}000
$$

accelerator-hours, before failed runs, evaluation, checkpoints, and recovery time.

## Choose a first layout

Start with eight-way tensor parallelism inside each node. This keeps frequent per-layer activation communication on the fast local links.

That leaves 64 data-parallel groups across nodes.

Before selecting full parameter sharding, check whether a lighter option fits. With eight-way tensor parallelism:

- weights per GPU: $140/8 = 17.5$ GB;
- gradients per GPU: another 17.5 GB;
- Adam moments sharded over the 64 data-parallel ranks: $560/(8 \times 64) \approx 1.1$ GB.

This ZeRO-1 style estimate uses about 36.1 GB per GPU for model state. It leaves about 44 GB for activations, temporary buffers, communication, and safety margin.

This may fit with activation checkpointing. If measured peak memory is safe, it avoids the parameter all-gathers required by full parameter sharding.

If activations still do not fit:

1. reduce the local micro-batch;
2. checkpoint activations;
3. use sequence or context parallelism for long sequences;
4. then consider parameter sharding or pipeline parallelism if the earlier steps are not enough.

Do not start with every parallelism dimension.

## Build the global batch

At sequence length 8,192, use one sequence per data-parallel replica per micro-batch and accumulate eight micro-batches.

$$
64 \times 1 \times 8 \times 8192
= 4{,}194{,}304\text{ tokens per optimizer step}.
$$

The 300-billion-token run needs about:

$$
300\text{B} / 4.194\text{M} \approx 71{,}500
$$

optimizer steps.

Tensor-parallel ranks work on the same sequences, so they do not multiply the global batch.

The target batch must also make sense for optimization. If it changes from the validated training recipe, retune learning rate, warmup, and possibly the token schedule.

## Check communication

Each tensor-parallel rank owns about 17.5 GB of gradients. Gradient synchronization may use an all-reduce or an equivalent reduce-scatter plus parameter all-gather for optimizer-state sharding. The ring traffic has this large-message lower bound:

$$
\frac{2(63/64)(17.5\text{ GB})}{45\text{ GB/s}} \approx 0.77\text{ seconds}.
$$

The real value depends on bucketing, topology, contention, and overlap with backpropagation.

Trace the last gradient buckets. A long reduction tail after backward compute ends is exposed communication and directly increases step time.

## Validate the plan in stages

### Stage 1: one-node correctness

- confirm loss and gradient behavior;
- verify parameter count and FLOP estimate;
- record peak memory by component;
- measure kernel shapes and numerical stability.

### Stage 2: multi-node scaling

Test a small grid, such as 8, 64, 128, and 512 GPUs when practical. Keep the global batch or report any change.

Record:

- tokens per second;
- MFU;
- median and tail step time;
- exposed collective time;
- per-rank imbalance;
- peak memory.

### Stage 3: reliability

- checkpoint model, optimizer, data position, and random state;
- test restart before the full run;
- validate checkpoint write time and storage load;
- define health checks for loss spikes, NaNs, stalled ranks, and slow nodes;
- keep a known-good checkpoint for rollback.

### Stage 4: convergence

Run enough tokens to test the learning curve, not only systems throughput. Compare with smaller-scale predictions and monitor held-out loss by data source.

## What an L4 answer sounds like

> "Use 512 GPUs, mixed precision, tensor parallelism, FSDP, activation checkpointing, and a distributed training framework."

This names tools without calculating fit, time, batch, or communication.

## What an L5 answer adds

An L5 candidate estimates 70 billion parameters, about 840 GB of unsharded model state, and the $1.26 \times 10^{23}$ parameter-matrix FLOP baseline. A stronger estimate adds about $1.93 \times 10^{22}$ FLOPs for long-context attention, giving about 6.6 days at 50% MFU. They place tensor parallelism inside each node and build the global batch correctly.

## What an L6 answer adds

An L6 candidate starts with the least expensive layout that fits. They notice that tensor parallelism plus optimizer-state sharding may fit without full parameter sharding. They reserve memory margin, estimate exposed gradient communication, preserve optimization behavior, and define a staged scaling, restart, and convergence plan.

They also report uncertainty. The 50% MFU and communication estimates are planning assumptions that traces must replace.

## Strong-hire signals

- Arithmetic comes before framework choice.
- Tensor-parallel ranks are not counted as extra data replicas.
- Model state and activations are budgeted separately.
- The topology determines group placement.
- Wall time and accelerator-hours are both reported.
- The candidate tests a simpler sharding stage before full parameter sharding.
- Communication overlap is measured, not assumed.
- Restart and convergence tests happen before the full run.

## Common follow-up

"Why not use full parameter sharding immediately?"

Full parameter sharding saves more memory. It also adds parameter all-gathers during forward and backward work. If a lighter sharding stage already leaves enough activation and buffer memory, the extra communication may not help. Measure both if the answer is close.

*Related: [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/), [strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/), [accelerator network topology](/concepts/accelerator-network-topology/), and [train a 100B parameter model](/questions/train-100b-model/).*