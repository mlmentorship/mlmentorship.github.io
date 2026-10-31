---
title: "Profiling distributed ML workloads"
description: "Use step traces, roofline limits, and communication timelines to find the exposed bottleneck in training and inference."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

Distributed workload profiling follows unfinished dependencies to the synchronized step boundary before selecting an optimization.

## Summary

Which dependency must finish before the next optimizer transition can begin?
Compare these hypothetical schedules, holding work and numerical results fixed.
Assume aligned clocks and no compute/network resource interference.

| Schedule | Compute interval, ms | Collective interval, ms | Completion, ms | Exposed tail, ms |
| --- | --- | --- | --- | --- |
| Overlap | 0..10 | 4..10 | 10 | 0 |
| Serialized | 0..10 | 10..16 | 16 | 6 |

Both schedules contain sixteen milliseconds of aggregate activity, but only one needs sixteen milliseconds of elapsed time.
The retained schematic also changes compute length; use the table, not that extra difference, to isolate overlap alone.

**Learning objective:** distinguish an operation's total duration from the exposed portion that extends synchronized step time.

<!-- visual:profiling-exposed-collective-tail -->
<figure class="learning-figure plot-panel" aria-labelledby="profiling-overlap-title">
	<p class="visual-kicker">Critical-path accounting</p>
	<p class="visual-title" id="profiling-overlap-title">Only the collective time exposed beyond useful computation extends the step</p><div class="visual-argument" data-visual-argument="profiling-exposed-collective-tail" data-argument-kind="bars" data-argument-composition="endpoint"><svg viewBox="0 0 360 394" role="img" aria-label="Only communication beyond useful overlap extends the step. A 40-unit collective overlaps 30 units of useful work. Its exposed tail is 10, not the full 40."><title>Only communication beyond useful overlap extends the step</title><desc>A 40-unit collective overlaps 30 units of useful work. Its exposed tail is 10, not the full 40.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Full collective duration</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="40" data-proof-min="0" data-proof-max="40" data-proof-zero="28" x="28" y="46" width="304" height="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><circle cx="332" cy="63.5" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M332 69L332 86" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="255.5" y="106" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="255.5" dy="0">40 toy time units</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Covered by useful computation</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="30" data-proof-min="0" data-proof-max="40" data-proof-zero="28" x="28" y="163" width="228" height="35" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></rect><circle cx="256" cy="180.5" r="4" style="fill:var(--viz-state-stroke);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><path d="M256 186L256 203" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><text x="255.5" y="223" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-state-stroke)"><tspan x="255.5" dy="0">30 toy time units</tspan></text><text x="18" y="257" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Exposed tail</tspan></text><path d="M28 298L332 298" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="10" data-proof-min="0" data-proof-max="40" data-proof-zero="28" x="28" y="280" width="76" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><circle cx="104" cy="297.5" r="4" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M104 303L104 320" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="104.5" y="340" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="104.5" dy="0">10 toy time units</tspan></text><text x="28" y="374" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="374" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">40</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-scroll">
		<svg viewBox="0 0 360 300" role="img" aria-labelledby="profiling-overlap-svg-title profiling-overlap-svg-desc">
			<title id="profiling-overlap-svg-title">Two step traces with the same collective duration but different exposed time</title>
			<desc id="profiling-overlap-svg-desc">The upper trace has a long backward-compute bar and a shorter rounded collective bar on a second lane. The collective finishes before backward compute, so it contributes no exposed tail. The lower trace has a shorter backward-compute bar followed by a collective of exactly the same drawn length as the upper collective. A bracket labels that collective's full duration as exposed, and its step boundary is later. Direct labels identify all bars, boundaries, and conclusions.</desc>
			<rect class="viz-plot-bg" x="76" y="28" width="254" height="236" rx="3"></rect>
			<path class="viz-gridline" d="M76 56H330 M76 128H330 M76 164H330 M76 236H330"></path>
			<text class="viz-callout" x="10" y="50">OVERLAPPED</text>
			<text class="viz-label" x="10" y="68">collective hidden</text>
			<text class="viz-axis-label" x="10" y="92">compute</text>
			<text class="viz-axis-label" x="10" y="120">network</text>
			<rect class="viz-node--input" x="108" y="76" width="144" height="20" rx="2"></rect>
			<text class="viz-callout" x="180" y="90" text-anchor="middle">backward compute</text>
			<rect class="viz-node--focus" x="164" y="104" width="80" height="20" rx="10"></rect>
			<text class="viz-callout" x="204" y="118" text-anchor="middle">collective</text>
			<path class="viz-axis" d="M76 136H252 M108 132V140 M164 132V140 M244 132V140 M252 128V144"></path>
			<text class="viz-label" x="108" y="153" text-anchor="middle">start</text>
			<text class="viz-label" x="252" y="153" text-anchor="middle">step end</text>
			<text class="viz-callout" x="322" y="118" text-anchor="end">exposed: 0</text>
			<text class="viz-callout" x="10" y="184">SERIALIZED</text>
			<text class="viz-label" x="10" y="202">collective exposed</text>
			<text class="viz-axis-label" x="10" y="226">compute</text>
			<text class="viz-axis-label" x="10" y="254">network</text>
			<rect class="viz-node--input" x="108" y="210" width="112" height="20" rx="2"></rect>
			<text class="viz-callout" x="164" y="224" text-anchor="middle">backward compute</text>
			<rect class="viz-node--focus" x="220" y="238" width="80" height="20" rx="10"></rect>
			<text class="viz-callout" x="260" y="252" text-anchor="middle">same collective</text>
			<path class="viz-operating-guide" d="M220 204V272 M300 204V272 M220 270H300"></path>
			<text class="viz-callout" x="260" y="202" text-anchor="middle">exposed tail: full duration</text>
			<path class="viz-axis" d="M76 272H300 M108 268V276 M220 268V276 M300 264V280"></path>
			<text class="viz-label" x="108" y="287" text-anchor="middle">start</text>
			<text class="viz-label" x="300" y="287" text-anchor="middle">step end</text>
		</svg>
	</div>
	<div class="visual-derivation-caption">the rounded network bars have equal duration. The upper collective finishes before compute; the lower collective extends completion. Compute lengths also differ here; the table isolates overlap with equal compute. This original schematic follows the <a href="https://docs.pytorch.org/docs/2.13/notes/ddp.html">DDP design note</a>, <a href="https://jax-ml.github.io/scaling-book/profiling/">JAX profiling guide</a>, and <a href="https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html">Nsight Systems guide</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="profiling-exposed-collective-tail"><strong>Check:</strong> Why can reducing equal-duration collectives have different effects on elapsed step time in the two traces?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="profiling-exposed-collective-tail">A 40-unit collective overlaps 30 units of useful work. Its exposed tail is 10, not the full 40.</span></figcaption>
</figure>

## Read the trace from the outside in

### 1. Mark the training or inference step

Let compute finish at $c$, a ready collective start at $s$, and its duration be $d$.
If both results are required, completion is $T=\max(c,s+d)$.
The network tail is $\max(0,s+d-c)$, not automatically $d$.
This simplified model excludes additional dependencies and interference.
Launching an asynchronous operation does not establish that its result is ready.

### 2. Find large idle gaps

Trace each wait back to input readiness, a producing kernel, another rank, or host synchronization.
If one rank's required bucket is ready at twelve ms, the collective cannot start at four ms.
With a six-ms collective, completion is at least eighteen ms in this simplified arrival model.
Inspect all ranks, not only the fastest backward pass.

### 3. Find the longest exposed operations

Moving the six-ms collective inside the ten-ms compute window saves six ms in the controlled example.
Halving a completely hidden collective saves zero ms there.
Resource contention can invalidate either prediction; measure the resulting compute duration too.

## Run controlled experiments

Keep shapes, batch, dtype, topology and timing scope fixed.
Check numerical outputs before accepting a speedup.
Record one hypothesis, change, predicted boundary movement, correctness result, measured median/tail, trace difference and keep/revert decision.
Retain unsuccessful trials.

### Change bucket readiness

**New task:** overlap the last bucket without using gradients before they exist.

**Picture and change:** move readiness from ten ms to four ms by changing bucket order or size.
The six-ms collective then ends at ten instead of sixteen ms, if compute remains unchanged.

**Check:** a bucket ready at nine ms ends at fifteen ms; the tail is still five ms.
An earlier API call without earlier data readiness does not produce the predicted gain.

**Use when:** the trace shows a genuinely exposed final communication bucket.

### Change tensor handoff

**New task:** remove a redundant gather/reshard between adjacent operations.

**Picture and change:** a serial handoff costs two ms around ten-ms useful work, giving twelve ms total.
A compatible producer/consumer layout removes that event, predicting ten ms before resource effects.

**Check:** two ranks holding additive partial results two and three represent five, not two.
Removing their reduction is incorrect, even if the trace becomes shorter.
Verify global outputs, communication semantics and temporary memory peaks.

**Use when:** layout conversion, rather than required mathematical reduction, extends the dependency path.

## Reconstruct the diagnosis

Why can a shorter local backward pass leave the synchronized step unchanged?

<details>
<summary>Reasoned answer</summary>

Another rank or required transfer may still finish later.
Compare input-ready, kernel and collective timestamps on aligned clocks.
Only movement of the completion dependency establishes a step-time improvement.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Why AI labs care

Low utilization can reflect input stalls, small kernels, memory traffic, network waits, pipeline bubbles or host work.
It does not identify a repair.

## Start with stable measurements

Verify correctness, exclude compilation/warmup and capture several representative steady steps.
Store software versions, topology, shapes, seed and profiler settings.
State whether median and p95 measure rank-local device work or synchronized wall-clock steps.
Check instrumentation overhead and clock alignment.

## Build three lower bounds

### Compute bound

Useful FLOPs divided by applicable peak FLOPs gives an optimistic compute-time floor.
State precision and the FLOP convention.

### Memory bound

Bytes moved divided by effective bandwidth bounds a specified kernel's memory time.
Roofline compares intensity against the compute/bandwidth ratio.

### Communication bound

Use latency plus per-device bytes divided by measured collective bandwidth for the matching message size and group.
Peak link bandwidth is insufficient.
These bounds may overlap; adding them is not a general elapsed-time bound.

### 4. Group kernels by type

Separate matmuls, attention, elementwise work, copies and collectives.
Tiny kernels suggest launch/fusion investigations; poor matmul shapes suggest a different experiment.

### 5. Compare ranks

Compare input lengths, arrivals, device clocks, network placement and thermal behavior.

## Common trace patterns

| Pattern | Discriminating check |
| --- | --- |
| Early idle gap | input-ready timestamp |
| Tiny kernels and gaps | launch overhead versus fusion |
| Late bucket | readiness and rank arrivals |
| Repeated reshard | producer/consumer layouts |
| Pipeline bubbles | stage balance and microbatches |

## Profiling a sharded program

Record layouts and identify inserted gathers, reduce-scatters, all-reduces, all-to-alls and host transfers.
Preserve the distinction between complete values and additive partial results.

## Profiling compilation

Match source operations and shapes to compiled fusions, device kernels, communications and allocation peaks.
One source operation need not correspond to one kernel.

## In an interview

Define the workload and metric, locate the completion dependency, predict one change, then measure correctness and timing together.

## Common mistakes

One fast warmup-contaminated sample is not a steady-state speedup.
Changing batch size confounds a kernel comparison.
Synthetic timelines establish arithmetic, not measured hardware performance.

*Related: [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/), [strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/), and [optimize an accelerator workload](/questions/optimize-accelerator-workload/). Further reading: [profiling in the JAX Scaling Book](https://jax-ml.github.io/scaling-book/profiling).*

</div>