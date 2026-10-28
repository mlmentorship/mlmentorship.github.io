---
title: "Design fault-tolerant distributed training"
description: "Classify fail-stop, straggler, state-divergence, and silent-corruption failures before choosing restart, elasticity, redundancy, or repair."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> A 2,048-GPU training job loses workers, stalls on collectives, and occasionally produces rank-local checksum mismatches. Design detection, recovery, and evidence preservation without corrupting the experiment.

Design recovery for distributed training that preserves one coherent experiment state despite worker, communication, or numerical failures.

## Problem

**Context.** A hypothetical 2,048-GPU job loses workers, stalls during collectives, and sometimes reports rank-local checksum mismatches.
Retries can duplicate or omit updates if ranks no longer agree on the committed step.

**Task.** Specify detection, failure classification, evidence preservation, checkpoint publication, and recovery scopes.
Explain how to prove that resumed ranks share compatible parameters, optimizer state, data position, and step identity.

**Constraints.** Distinguish idempotent work from externally committed updates and partial checkpoint writes.
Preserve diagnostic evidence, define rollback authority, and do not resume solely because a replacement worker starts successfully.

**Learning objective:** Choose the narrowest safe recovery scope by testing whether ranks share one proven training state before retrying, rebuilding a group, or rolling back.

<!-- visual:distributed-recovery-consistency-gate -->
<figure class="learning-figure editorial-illustration" aria-labelledby="distributed-recovery-consistency-gate-edition-title"><p class="visual-title" id="distributed-recovery-consistency-gate-edition-title">A live rank is not proof of a coherent training state</p><svg viewBox="0 0 360 454" role="img" aria-labelledby="distributed-recovery-consistency-gate-edition-svg-title distributed-recovery-consistency-gate-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="distributed-recovery-consistency-gate-edition-svg-title">A live rank is not proof of a coherent training state</title><desc id="distributed-recovery-consistency-gate-edition-svg-desc">An explicitly illustrative comparison shows a mixed state with rank zero at step 100 and rank one at step 101. Restarting one rank alone leaves that mismatch. The contrasting rollback restores model, optimizer, RNG, and data position from committed checkpoint 98 across the consistency domain. Supported narrow recovery remains valid only when it proves the same state contract.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative mixed state: rank 0 = 100;</tspan><tspan x="180" dy="19">rank 1 = 101</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">RESTART ONE</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">RESTORE DOMAIN</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Rank 0 model / optimizer step</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">100 / 100</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">98 / 98</tspan></text></g><path d="M15 207L345 207" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="227" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Rank 1 model / optimizer step</tspan></text><g data-editorial-node><rect x="15" y="241" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="90" y="279.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">101 / 101</tspan></text></g><g data-editorial-node><rect x="195" y="241" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="279.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">98 / 98</tspan></text></g><path d="M15 323L345 323" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="343" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">RNG + data position + integrity</tspan></text><g data-editorial-node><rect x="15" y="357" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="90" y="386" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">not proved</tspan><tspan x="90" dy="19">coherent</tspan></text></g><g data-editorial-node><rect x="195" y="357" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="386" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">same committed</tspan><tspan x="270" dy="19">98</tspan></text></g></svg><figcaption><strong>Read it this way:</strong> A single-rank restart leaves the pictured mismatch. Restore and validate the consistency domain. Narrow recovery is valid only when its supported protocol proves coherence. <span class="visual-reader-check" data-reader-check="distributed-recovery-consistency-gate"><strong>Check:</strong> Why is a healthy restarted worker insufficient when ranks disagree about optimizer or data state?</span></figcaption></figure>

## Four failure classes

### Fail-stop

A worker, process, node, or link fails visibly. Synchronous collectives generally require every rank in the group; one missing participant causes timeout or abort. Recovery usually restarts a consistent group from a complete checkpoint, unless the algorithm and runtime support elastic membership.

### Straggler

A rank remains alive but arrives late because of hardware degradation, data skew, checkpoint I/O, network contention, or thermal throttling. The slowest rank sets step time. Restarting the whole job may hide the symptom without identifying the bad component.

### State divergence

Ranks disagree on step, RNG, data position, optimizer state, or parameters. A skipped collective or partial restore can hang immediately or silently change training. Checkpoint completeness and cross-rank state validation matter more than process liveness.

### Silent corruption

A transfer or computation produces wrong values without an immediate error. Checksums, finite checks, range checks, redundant critical calculations, and anomaly detection provide evidence. Replication helps only when failures are sufficiently independent and the comparison itself is trustworthy.

## The recovery contract

A checkpoint is recoverable only if it defines:

- model, optimizer, scheduler, scaler, and RNG state;
- global step and consumed data position;
- sharding and parallel-group metadata;
- code, configuration, data, and environment versions;
- completeness and integrity checks;
- a commit protocol that cannot expose a half-written checkpoint as valid.

For large sharded checkpoints, each shard can write to a temporary location, report integrity metadata, then publish one manifest only after every required shard succeeds.

## What an L4 answer sounds like

> "Checkpoint every few minutes, detect failed GPUs, replace them, and restart. Use retries for network errors."

The answer treats all failures as visible and independent. It does not define consistent state, collective semantics, data replay, or corruption detection.

## What an L5 answer adds

An L5 candidate separates failure classes, defines timeouts and health signals, uses atomic checkpoint manifests, and restarts from a known-consistent state. They explain the cost trade-off for checkpoint interval:

- frequent checkpoints reduce lost compute but consume bandwidth and storage;
- infrequent checkpoints improve throughput but increase expected rollback;
- asynchronous checkpointing reduces pause time but adds memory pressure and consistency complexity.

They also instrument collective duration by rank and topology, checkpoint phases, data-loader position, hardware errors, and cross-rank checksums.

## What an L6 answer adds

An L6 candidate protects scientific validity. Replaying data, changing world size, replacing hardware, or switching kernels can change ordering and numerical behavior. The system records lineage and determines which changes preserve comparability.

They choose recovery scope from failure scope:

- retry a read only if the operation is idempotent and state remains aligned;
- rebuild a communication group after a fail-stop only if membership change is supported by the training decomposition;
- restart the affected parallel replica when state boundaries permit;
- restart the full job when optimizer or model consistency is uncertain;
- quarantine a device or link when recurrence follows topology.

They also design drills. A failure system trusted only after real incidents is untested infrastructure. Inject worker exits, delayed ranks, corrupt manifests, full storage, and network partitions in small canary jobs.

## Tells that get you a strong-hire vote

- You classify fail-stop, straggler, divergence, and corruption separately.
- Collective participation and timeout semantics are explicit.
- A checkpoint includes optimizer, RNG, data position, and sharding metadata.
- Half-written checkpoints cannot appear valid.
- Recovery scope follows state-consistency evidence.
- Data replay and world-size changes enter experiment lineage.
- Fault injection validates the design before a frontier run.

## Tells that get you down-leveled

- "Retry three times" without idempotency or state alignment.
- Checkpointing only model weights.
- Treating a slow rank as a failed rank.
- Assuming elastic membership preserves algorithm behavior automatically.
- No silent-corruption detection.
- Declaring recovery when the job resumes but the trajectory changed invisibly.

## Common follow-up

"Can we continue with 2,047 GPUs after one worker fails?"

Only if the parallel decomposition, optimizer semantics, batch construction, communication groups, and checkpoint state support a membership change. Tensor and pipeline partitions often require fixed group shapes. Even pure data parallelism changes global batch or per-rank work unless reconfigured. The right answer is conditional, not automatically elastic.

*Related: [fault-tolerant collectives](/concepts/fault-tolerant-collectives/), [all-reduce and collectives](/concepts/all-reduce-and-collectives/), and [train a 100B parameter model](/questions/train-100b-model/).*
