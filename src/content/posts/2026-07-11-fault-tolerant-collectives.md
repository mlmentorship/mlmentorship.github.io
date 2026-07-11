---
title: "Fault-tolerant collectives"
description: "What happens when a rank fails, stalls, diverges, or corrupts data inside all-reduce, and which guarantees recovery can actually preserve."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A fault-tolerant collective detects missing, delayed, divergent, or corrupted participants and either completes with a stated guarantee or aborts into a consistent recovery path. Ordinary all-reduce assumes the group is healthy; fault tolerance begins when that assumption fails.

## Why it matters

Synchronous training advances at the pace of the slowest required rank. One fail-stop worker can abort the communication group, one straggler can stall every peer, and one rank with divergent state can produce a numerically valid but scientifically invalid update.

"Retry the collective" is not a general solution. A retry is safe only when every participant agrees on the operation, input state, and whether the previous attempt committed.

## Failure model first

| Failure | Observable symptom | Core risk |
| --- | --- | --- |
| Fail-stop rank | timeout, connection close, process exit | collective cannot complete |
| Straggler | long tail in operation duration | global step stalls |
| Rank desynchronization | mismatched sequence number or hang | ranks execute different collectives |
| Data corruption | checksum or range mismatch | wrong aggregate may look valid |
| Network partition | subgroups remain alive | split progress or indefinite wait |
| Byzantine participant | arbitrary or adversarial value | aggregate integrity fails |

Fail-stop tolerance does not imply Byzantine tolerance. Most ML communication stacks detect transport and process failures but do not defend against an arbitrary malicious gradient.

## Recovery choices

### Abort and restart

The common robust path is to abort the process group and restart a consistent job state from checkpoint. This costs rollback but avoids ambiguous partial completion.

### Elastic membership

A runtime may rebuild the group with fewer or replacement workers. The algorithm must tolerate changed world size, batch partition, optimizer semantics, and parallel-group shape. Tensor and pipeline parallelism often constrain membership more tightly than data parallelism.

### Redundancy

Duplicate computation, parity-like coding, redundant parameter shards, or multiple gradient estimates can reconstruct some failures. Redundancy consumes compute or bandwidth and relies on an independence assumption.

### Approximate aggregation

Asynchronous or partial aggregation can continue with missing workers, but it changes the optimization algorithm. Staleness, bias, effective batch, and convergence become part of the contract. Availability is not free correctness.

## Detection

Useful signals include:

- collective sequence numbers and deadlines;
- rank-local start and finish timestamps;
- tensor shape, dtype, finite checks, norms, and checksums;
- hardware and link errors;
- heartbeat and process liveness;
- parameter or optimizer checksums at selected boundaries;
- checkpoint-manifest completeness.

A timeout detects absence, not cause. Preserve rank-local evidence before the group tears down.

## Silent corruption

Checksums can detect transmission or storage corruption, but a deterministic checksum does not prove the computation itself was correct. Range checks and redundant calculation catch different failures. Robust aggregation methods such as coordinate-wise median or trimmed means address outliers under specific assumptions, but can distort ordinary stochastic gradients and scale poorly.

State the adversary and statistical assumptions before calling an aggregate robust.

## Recovery and exactly-once language

Distributed training rarely needs transaction-style exactly-once execution of a collective. It needs a consistent optimizer transition. If some ranks applied step $t$ and others did not, replaying communication alone is insufficient. Restore all required state or use a commit protocol that prevents partial publication of the step.

## Common confusions

- **"NCCL retries failed collectives."** A failed process group commonly requires teardown and reconstruction; transport retry does not restore rank state.
- **"Elastic means fault tolerant."** Elastic membership solves one recovery mechanism, not state consistency or algorithm equivalence.
- **"Averaging removes bad gradients."** Averaging reduces independent zero-mean noise. Correlated or adversarial errors do not vanish as $1/n$.
- **"Checkpointing solves corruption."** A corrupted or incomplete checkpoint can preserve the failure unless integrity and completeness are verified.
- **"Asynchronous training is always more available."** It trades synchronization for staleness and different convergence behavior.

## In an interview

Start with failure classes, collective semantics, and consistent state. Then discuss timeouts, evidence, checkpoint commit, recovery scope, and whether the recovered job is still the same experiment.

*Related: [all-reduce and collectives](/concepts/all-reduce-and-collectives/), [FSDP and ZeRO](/concepts/fsdp-and-zero/), and [design fault-tolerant distributed training](/questions/design-fault-tolerant-distributed-training/).*
