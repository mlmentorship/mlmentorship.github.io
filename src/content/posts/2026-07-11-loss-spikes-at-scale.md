---
title: "Loss spikes at scale"
description: "A training spike is a first-bad-transition problem across data, numerics, optimizer, distributed state, software, and hardware."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A loss spike is a sudden, material increase in training loss or related instability that may recover, persist, or cascade into divergence. At scale, diagnosis requires the last good state, first bad transition, and rank-local evidence.

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

**Learning objective:** classify a loss event as transient, persistent, divergent, or metric-only from its trace without mistaking the trace shape for a root cause.

<!-- visual:loss-spike-trace-taxonomy -->
<figure class="learning-figure plot-panel" aria-labelledby="loss-spike-taxonomy-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="loss-spike-taxonomy-title">What does the trace establish, and what remains unknown?</p>
	<svg viewBox="0 0 360 400" role="img" aria-labelledby="loss-spike-taxonomy-svg-title loss-spike-taxonomy-svg-desc">
		<title id="loss-spike-taxonomy-svg-title">Four qualitative signatures of a training loss event</title>
		<desc id="loss-spike-taxonomy-svg-desc">Four aligned plots run left to right in time. A transient spike rises sharply and returns to its prior trend. A persistent spike rises and settles at a worse plateau. A divergent trace rises and then accelerates toward failure. In the metric-only panel, a solid training-loss trace stays level while a dashed evaluation-metric trace jumps. The shapes classify symptoms but do not identify whether data, numerics, optimization, distributed state, software, or hardware caused them.</desc>
		<g aria-label="Transient spike returns to the prior trend">
			<rect class="viz-plot-bg" x="0" y="0" width="360" height="92" rx="4"></rect>
			<text class="viz-axis-label" x="12" y="25">TRANSIENT</text>
			<text class="viz-label" x="12" y="45">returns to prior trend</text>
			<path class="viz-axis" d="M128 15V74H344"></path>
			<path class="viz-baseline" d="M128 65H344"></path>
			<path d="M130 65L162 63L188 27L214 60L246 66L280 63L312 64L342 61" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path>
			<text class="viz-label" x="334" y="57" text-anchor="end">recovered</text>
		</g>
		<g aria-label="Persistent spike settles at a worse loss level">
			<rect class="viz-plot-bg" x="0" y="101" width="360" height="92" rx="4"></rect>
			<text class="viz-axis-label" x="12" y="126">PERSISTENT</text>
			<text class="viz-label" x="12" y="146">new, worse plateau</text>
			<path class="viz-axis" d="M128 116V175H344"></path>
			<path class="viz-baseline" d="M128 166H344"></path>
			<path d="M130 166L162 164L188 126L214 134L246 132L280 136L312 133L342 135" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path>
			<text class="viz-label" x="334" y="126" text-anchor="end">shifted regime</text>
		</g>
		<g aria-label="Divergent loss compounds toward failure">
			<rect class="viz-plot-bg" x="0" y="202" width="360" height="92" rx="4"></rect>
			<text class="viz-axis-label" x="12" y="227">DIVERGENT</text>
			<text class="viz-label" x="12" y="247">instability compounds</text>
			<path class="viz-axis" d="M128 217V276H344"></path>
			<path class="viz-baseline" d="M128 267H344"></path>
			<path d="M130 267L162 265L188 252L214 246L246 233L280 225L312 213L342 204" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path>
			<text class="viz-label" x="334" y="220" text-anchor="end">toward failure</text>
		</g>
		<g aria-label="Metric-only event where training loss stays stable while an evaluation metric changes">
			<rect class="viz-plot-bg" x="0" y="303" width="360" height="92" rx="4"></rect>
			<text class="viz-axis-label" x="12" y="328">METRIC-ONLY</text>
			<text class="viz-label" x="12" y="348">training state stays healthy</text>
			<path class="viz-axis" d="M128 318V377H344"></path>
			<path d="M130 365L164 363L198 365L232 362L266 364L300 363L342 365" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2.5;stroke-linecap:round"></path>
			<path d="M130 349L164 350L198 349L216 324L250 326L286 323L342 325" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-dasharray:5 4;stroke-linecap:round;stroke-linejoin:round"></path>
			<text class="viz-label" x="337" y="318" text-anchor="end">eval metric · dashed</text>
			<text class="viz-label" x="337" y="378" text-anchor="end">training loss · solid</text>
		</g>
		<text class="viz-label" x="344" y="397" text-anchor="end">time → · illustrative, not to scale</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> scan each trace from left to right. Returning to the prior trend is transient; settling higher is a persistent regime change; accelerating upward is divergent; and a changed evaluation metric beside stable training loss is metric-only. These shapes classify the symptom, not its cause. Preserve the last good state and test data, numerics, optimization, distributed state, software, and hardware separately. Original qualitative synthesis informed by <a href="https://arxiv.org/abs/2204.02311">the PaLM training report</a>, <a href="https://arxiv.org/abs/2312.16903">Takase et al. (2024)</a>, and <a href="https://arxiv.org/abs/2407.21783">the Llama 3 system report</a>.</figcaption>
</figure>

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
