---
title: "Loss spikes at scale"
description: "A training spike is a first-bad-transition problem across data, numerics, optimizer, distributed state, software, and hardware."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A loss spike is a sudden, material increase in training loss or related instability that may recover, persist, or cascade into divergence. At scale, diagnosis requires the last good state, first bad transition, and rank-local evidence.

## Why it matters

Large training runs expose rare samples, long numerical chains, distributed synchronization, changing data mixtures, and hardware faults. A scalar loss curve compresses all of those systems into one symptom.

Gradient clipping or a lower learning rate can make a run continue without identifying whether the cause was data, numerical overflow, a bad resume, rank divergence, or a kernel regression.

## Spike taxonomy

### Transient, self-recovering

One unusual batch creates a large update but the optimizer returns to the prior trend. Even if capability recovers, repeated spikes can waste compute or signal a fragile margin.

### Persistent regime change

Loss settles at a worse level after the event. The optimizer state, parameters, data distribution, or schedule may have crossed into a new regime.

### Divergent

Loss, norms, or non-finite values compound until recovery from checkpoint is required.

### Metric-only

The training state is healthy but logging, evaluation preprocessing, or a specific domain metric changes. A domain-specific spike can reveal evaluation mismatch rather than model failure.

## Causal families

- **Data:** malformed tokens, packing or masking defects, extreme lengths, mixture shift, duplicates, corrupt labels.
- **Numerics:** overflow, underflow, unstable softmax or normalization, precision change, bad loss scaling.
- **Optimization:** learning-rate discontinuity, missing clipping, corrupted moments, accumulation change, bad scheduler resume.
- **Distributed state:** rank desynchronization, partial restore, inconsistent batch or RNG, collective corruption.
- **Software:** kernel, compiler, framework, model-code, or configuration change.
- **Hardware:** device memory errors, network faults, degraded node, storage corruption.

## Investigation

1. Save the last good and first bad checkpoints and batch identities.
2. Compare global and rank-local loss, activation norms, gradient norms, scaler state, and learning rate.
3. Find the first layer or tensor with abnormal or non-finite state.
4. Replay the batch and checkpoint under controlled precision and kernels.
5. Compare data, code, configuration, topology, and environment around the transition.
6. Test one causal family at a time.
7. Recover from known-good state with the smallest justified change.
8. Verify that recovery returns to the expected trajectory and record any regime change.

## Mitigations

Gradient clipping bounds update magnitude. Warmup controls early optimizer transients. BF16 avoids FP16's narrow exponent range. FP32 reductions or softmax can protect sensitive operations. QK normalization, embedding normalization, initialization, and residual scaling can improve margin in some architectures.

These are design choices, not universal cures. Apply a mitigation because evidence supports the failure mechanism.

## Monitoring

Track distributions and rank-local tails:

- activation and gradient norms by layer;
- non-finite tensor location;
- update-to-weight ratio;
- optimizer moment norms;
- loss by domain and sequence-length bucket;
- token and mask statistics;
- scaler and skipped-step history;
- collective timing and rank checksums;
- hardware and software event timeline.

Average loss can remain normal while one layer, rank, or data slice becomes unstable.

## Common confusions

- **"Every spike is bad data."** Data is one family among several.
- **"Clipping fixes the root cause."** It bounds one symptom and can hide recurrence.
- **"A replay on one GPU clears the batch."** Distributed timing, precision, or topology may be necessary for failure.
- **"The run recovered, so science is unchanged."** A changed checkpoint, filter, kernel, or schedule can alter comparability.
- **"Only NaNs matter."** Persistent norm or loss shifts can damage training while remaining finite.
- **"Global metrics are enough."** Reduction can hide rank-local onset.

## In an interview

Preserve evidence, classify the spike, isolate the first bad transition, compare rank-local state, reproduce cheaply, separate recovery from root cause, and state whether the intervention changes experiment validity.

*Related: [debug a frontier LLM training run](/questions/debug-frontier-llm-training-run/), [gradient clipping](/concepts/gradient-clipping/), and [floating-point formats](/concepts/floating-point-formats/).*
