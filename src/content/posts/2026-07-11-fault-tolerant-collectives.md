---
title: "Fault-tolerant collectives"
description: "What happens when a rank fails, stalls, diverges, or corrupts data inside all-reduce, and which guarantees recovery can actually preserve."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A fault-tolerant collective detects missing, delayed, divergent, or corrupted participants and either completes with a stated guarantee or aborts into a consistent recovery path. Ordinary all-reduce assumes the group is healthy; fault tolerance begins when that assumption fails.

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

<!-- visual:collective-failure-commit-boundary -->
<figure class="learning-figure plot-panel" aria-labelledby="collective-commit-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="collective-commit-title">Why is retrying the collective unsafe after a partial optimizer commit?</p>
	<svg viewBox="0 0 360 500" role="img" aria-labelledby="collective-commit-svg-title collective-commit-svg-desc">
		<title id="collective-commit-svg-title">Unsafe collective replay compared with consistent checkpoint recovery</title>
		<desc id="collective-commit-svg-desc">All four ranks start step t from the same committed checkpoint at step t minus one. During the failed attempt, ranks zero and one apply optimizer step t, rank two fails, and rank three has not applied it. Retrying only the collective from those mixed states can make ranks zero and one apply the update twice while rank three applies it once, so the job diverges. The safe path aborts the old group, restores model, optimizer, scheduler, scaler, RNG, and data position for every rank from the same committed checkpoint, creates a new group, and replays the whole step once.</desc>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="488" rx="6"></rect>
		<rect class="viz-node viz-node--output" x="38" y="18" width="284" height="50" rx="5"></rect>
		<text class="viz-node-label" x="180" y="38">Committed checkpoint · step t−1</text>
		<text class="viz-node-value" x="180" y="56">same model + optimizer + input position on every rank</text>
		<path d="M180 68V88" style="stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M174 84L180 92L186 84Z" style="fill:var(--viz-edge)"></path>
		<text class="viz-axis-label" x="180" y="106" text-anchor="middle">FAILED ATTEMPT AT STEP t</text>
		<rect class="viz-node viz-node--focus" x="12" y="118" width="78" height="58" rx="5"></rect>
		<text class="viz-node-value" x="51" y="138">RANK 0</text>
		<text class="viz-node-label" x="51" y="158">applied t</text>
		<text class="viz-label" x="51" y="171" text-anchor="middle">state Sₜ</text>
		<rect class="viz-node viz-node--focus" x="98" y="118" width="78" height="58" rx="5"></rect>
		<text class="viz-node-value" x="137" y="138">RANK 1</text>
		<text class="viz-node-label" x="137" y="158">applied t</text>
		<text class="viz-label" x="137" y="171" text-anchor="middle">state Sₜ</text>
		<rect class="viz-node" x="184" y="118" width="78" height="58" rx="5" style="stroke-dasharray:6 4"></rect>
		<text class="viz-node-value" x="223" y="138">RANK 2</text>
		<text class="viz-node-label" x="223" y="158">failed</text>
		<text class="viz-label" x="223" y="171" text-anchor="middle">no result</text>
		<rect class="viz-node" x="270" y="118" width="78" height="58" rx="5"></rect>
		<text class="viz-node-value" x="309" y="138">RANK 3</text>
		<text class="viz-node-label" x="309" y="158">not applied</text>
		<text class="viz-label" x="309" y="171" text-anchor="middle">state Sₜ₋₁</text>
		<path d="M180 176V199M180 199H91V216M180 199H269V216" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M85 212L91 220L97 212ZM263 212L269 220L275 212Z" style="fill:var(--viz-edge)"></path>
		<text class="viz-axis-label" x="91" y="236" text-anchor="middle">RETRY ONLY COMMUNICATION</text>
		<rect class="viz-node viz-node--focus" x="12" y="248" width="158" height="92" rx="5" style="stroke-dasharray:6 4"></rect>
		<text class="viz-callout" x="91" y="271" text-anchor="middle">Unsafe mixed-state replay</text>
		<text class="viz-node-value" x="91" y="292">rank 0/1 may apply t twice</text>
		<text class="viz-node-value" x="91" y="308">rank 3 applies t once</text>
		<text class="viz-callout" x="91" y="328" text-anchor="middle">states diverge</text>
		<text class="viz-axis-label" x="269" y="236" text-anchor="middle">RECOVER THE TRANSITION</text>
		<rect class="viz-node viz-node--output" x="190" y="248" width="158" height="92" rx="5"></rect>
		<text class="viz-callout" x="269" y="271" text-anchor="middle">Safe coordinated rollback</text>
		<text class="viz-node-value" x="269" y="292">abort old group</text>
		<text class="viz-node-value" x="269" y="308">restore every rank to Sₜ₋₁</text>
		<text class="viz-callout" x="269" y="328" text-anchor="middle">states agree</text>
		<path d="M269 340V358" style="stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M263 354L269 362L275 354Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--input" x="190" y="366" width="158" height="66" rx="5"></rect>
		<text class="viz-node-label" x="269" y="387">New process group</text>
		<text class="viz-node-value" x="269" y="405">replay forward + backward</text>
		<text class="viz-node-value" x="269" y="419">+ collective + optimizer once</text>
		<path d="M269 432V448" style="stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M263 444L269 452L275 444Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--output" x="190" y="456" width="158" height="28" rx="14"></rect>
		<text class="viz-node-label" x="269" y="475">all ranks at Sₜ</text>
		<path d="M91 340V470H174" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-dasharray:5 4"></path>
		<path d="M170 464L178 470L170 476Z" style="fill:var(--viz-focus-stroke)"></path>
		<text class="viz-label" x="22" y="463">cannot safely join</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> a collective result is not the commit boundary; the optimizer transition is. Once ranks disagree about whether step <code>t</code> was applied, replaying only communication can apply the update a different number of times. Abort the failed group, restore every required state component from one committed boundary, rebuild membership, and replay the whole step exactly once.</figcaption>
</figure>

## Common confusions

- **"NCCL retries failed collectives."** A failed process group commonly requires teardown and reconstruction; transport retry does not restore rank state.
- **"Elastic means fault tolerant."** Elastic membership solves one recovery mechanism, not state consistency or algorithm equivalence.
- **"Averaging removes bad gradients."** Averaging reduces independent zero-mean noise. Correlated or adversarial errors do not vanish as $1/n$.
- **"Checkpointing solves corruption."** A corrupted or incomplete checkpoint can preserve the failure unless integrity and completeness are verified.
- **"Asynchronous training is always more available."** It trades synchronization for staleness and different convergence behavior.

## In an interview

Start with failure classes, collective semantics, and consistent state. Then discuss timeouts, evidence, checkpoint commit, recovery scope, and whether the recovered job is still the same experiment.

*Related: [all-reduce and collectives](/concepts/all-reduce-and-collectives/), [FSDP and ZeRO](/concepts/fsdp-and-zero/), and [design fault-tolerant distributed training](/questions/design-fault-tolerant-distributed-training/).*
