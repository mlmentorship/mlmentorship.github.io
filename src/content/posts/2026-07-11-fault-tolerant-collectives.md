---
title: "Fault-tolerant collectives"
description: "What happens when a rank fails, stalls, diverges, or corrupts data inside all-reduce, and which guarantees recovery can actually preserve."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

Fault-tolerant training must restore a coherent optimizer transition, not merely complete a retried collective.

## Summary

Do all ranks agree on the operation, input state and whether its optimizer update already happened?
Consider two synthetic ranks starting with weight $w=1$.
The intended SGD update subtracts $0.1\cdot2=0.2$ using a fixed gradient.

| Boundary | Rank zero | Rank one |
| --- | --- | --- |
| Committed start | 1 | 1 |
| Partial application, then failure | 0.8 | 1 |
| Blindly apply again | 0.6 | 0.8 |
| Coordinated restore | 1 | 1 |
| Replay from restored inputs | 0.8 | 0.8 |

Replaying communication cannot undo an update already applied on one rank.
Even equal weights are insufficient for Adam if moments or step counters disagree.

## Failure model first

| Failure | Observable symptom | Core risk |
| --- | --- | --- |
| Fail-stop rank | timeout, connection close, process exit | collective cannot complete |
| Straggler | long tail in operation duration | global step stalls |
| Rank desynchronization | mismatched sequence number or hang | ranks execute different collectives |
| Data corruption | checksum or range mismatch | wrong aggregate may look valid |
| Network partition | subgroups remain alive | split progress or indefinite wait |
| Byzantine participant | arbitrary or adversarial value | aggregate integrity fails |

Fail-stop detection does not provide Byzantine tolerance against arbitrary participant values.
A timeout establishes missing timely completion, not the failure's cause.

## Recovery and exactly-once language

Abort the old group and restore all participants from one verified, committed checkpoint manifest.
Restore parameters, optimizer, scheduler, scaler, random-generator state, input position and step identity before rebuilding supported membership.
Record checkpoint lineage and replayed inputs.
The diagram's once-only result means one coherent intended transition under that protocol, not universal exactly-once execution.

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
	<figcaption><strong>Read it this way:</strong> four ranks disagree about whether update <code>t</code> happened. The left path repeats work from mixed states; the right path restores one complete boundary first. Replay is coherent only under the stated state-restoration and membership protocol.</figcaption>
</figure>

## Change the recovery contract

### Continue with fewer ranks

**New task:** resume after losing one data-parallel worker.

**Picture and change:** equal-sized local gradients $2,4,12$ previously averaged to six.
The surviving gradients two and four average to three, not six.

**Check:** dividing their sum by the old world size gives two, which is not even the surviving-batch mean.
Recompute normalization and effective batch; restore compatible state and record the changed experiment.
A missing required tensor shard needs reconstruction, not fewer-gradient averaging.

**Use when:** the runtime supports membership rebuild and the model's decomposition remains feasible.

### Admit an arbitrary participant

**New task:** a live rank can submit a finite but wrong gradient.

**Picture and change:** gradients two and one hundred average to fifty-one.
Both can pass shape, finiteness and byte-integrity checks.

**Check:** adding another honest value two makes the median two, but changes the aggregation rule.
That rule needs an explicit honest-participant and statistical model; heterogeneous legitimate gradients can also be suppressed.

**Use when:** the threat exceeds fail-stop transport detection.
Separate integrity checks, redundant computation and statistically robust aggregation.

## Reconstruct the boundary

Can matching checksums establish that the restored gradient is correct?

<details>
<summary>Reasoned answer</summary>

No. Matching bytes can encode the same wrong computation.
Check agreement and checkpoint completeness separately from numerical correctness and experiment equivalence.
Equal weights with unequal optimizer moments can still produce different next updates.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Recovery choices

### Abort and restart

Restore a committed consistency domain, not each rank's newest independently written file.
An incomplete or corrupted checkpoint can preserve the original failure.

### Elastic membership

Check data partition, loss denominator, optimizer state redistribution and tensor/pipeline group constraints.
Availability does not imply the original trajectory is preserved.

### Redundancy

Replicas, coded computation and redundant shards consume resources and require a specified reconstruction rule.
Common-mode faults can defeat supposedly independent copies.

### Approximate aggregation

Partial or asynchronous progress changes which gradients enter an update and their staleness.
Treat convergence and bias as new obligations, not free availability.

## Detection

Capture process-group identity, operation sequence, shape/dtype, rank starts/ends, deadlines, hardware errors and liveness.
Retain parameter/optimizer/step checks and checkpoint-manifest completeness.
Preserve evidence before teardown.
Transport retries cannot restore application state.

## Silent corruption

Checksums detect specified byte changes; redundant calculation tests different faults.
Neither guarantees arbitrary computational correctness.
Trimmed means and coordinate medians need stated assumptions and can distort legitimate stochastic gradients.

## Common confusions

Independent zero-mean noise may average down; correlated or adversarial errors need not.
Do not confuse transport completion with global optimizer publication.
Framework/version-specific group reconstruction is not a portable blanket retry promise.

## In an interview

State the failure model, committed boundary and recovery consistency domain.
Then specify detection, validation, supported membership and lineage before claiming the experiment survived unchanged.

*Related: [all-reduce and collectives](/concepts/all-reduce-and-collectives/), [FSDP and ZeRO](/concepts/fsdp-and-zero/), and [design fault-tolerant distributed training](/questions/design-fault-tolerant-distributed-training/).*

</div>
