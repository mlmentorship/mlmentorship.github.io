---
title: "Optimize an accelerator workload from a trace"
description: "Performance engineering is a measured argument: identify the bottleneck, predict the effect, preserve correctness, and shorten the critical path."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> You receive a correct but slow kernel on a simulated accelerator with scratchpad memory, SIMD lanes, multiple execution units, and a cycle trace. Improve it under a fixed time limit.

Do not optimize source code by aesthetic instinct. Establish a verified baseline, identify the limiting resource, predict which change moves it, and measure. The trace is the evidence that connects code to cycles.

## Start with a bottleneck model

Classify the workload along four axes:

1. **Work:** how many useful operations are required?
2. **Traffic:** how many bytes move through each memory level?
3. **Parallelism:** which operations are independent enough for SIMD, multiple units, or cores?
4. **Dependency depth:** what is the shortest possible critical path even with unlimited units?

A roofline-style check asks whether arithmetic throughput or memory bandwidth bounds performance:

$$
\text{attainable throughput} \leq \min(\text{peak compute},\ \text{bandwidth} \times \text{arithmetic intensity}).
$$

On a VLIW-like simulator, instruction packing and dependencies add another ceiling. More instructions per cycle help only when useful operations are ready and the required units are free.

## Read the trace in this order

- Long idle periods: dependency, scheduling, or unavailable data?
- Saturated unit: which operation class owns the critical path?
- Memory stalls: bandwidth, latency, bank conflict, or poor reuse?
- SIMD occupancy: are lanes doing useful work or masked out?
- Scratchpad use: can a larger tile fit without spilling?
- Repeated scalar work: can it be hoisted, vectorized, or precomputed?
- Load and compute overlap: are independent transfers hidden behind arithmetic?

State one hypothesis and predicted direction before changing code. "Vectorize" is not a hypothesis. "Packing eight independent comparisons into one SIMD instruction should remove roughly seven scalar compare slots from each tree level, unless gathers dominate" is.

<!-- visual:accelerator-critical-path-trace -->
<figure class="learning-figure plot-panel" aria-labelledby="accelerator-trace-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="accelerator-trace-title">Which optimization actually shortens the dependent chain?</p>
	<svg viewBox="0 0 360 382" role="img" aria-labelledby="accelerator-trace-svg-title accelerator-trace-svg-desc">
		<title id="accelerator-trace-svg-title">Three accelerator traces compare removing off-path work with shortening the critical path</title>
		<desc id="accelerator-trace-svg-desc">All rows use the same zero-to-twelve-cycle scale. In the baseline, a four-cycle load, four-cycle dependent compute operation, and four-cycle dependent store form a solid critical chain ending at cycle twelve. A dashed three-cycle helper overlaps the load and is not on that chain. Experiment A removes the helper, reducing instruction work but retaining the same twelve-cycle critical chain. Experiment B retains the helper but shortens the exposed load to two cycles, so dependent compute runs from cycles two to six and the store from six to ten, ending two cycles earlier.</desc>
		<defs><marker id="accelerator-trace-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path></marker></defs>
		<text class="viz-axis-label" x="18" y="25">cycle</text>
		<path class="viz-axis" d="M88 24H328"></path>
		<path class="viz-gridline" d="M88 20V372 M128 20V372 M168 20V372 M208 20V372 M248 20V372 M288 20V372 M328 20V372"></path>
		<text class="viz-label" x="88" y="15" text-anchor="middle">0</text><text class="viz-label" x="168" y="15" text-anchor="middle">4</text><text class="viz-label" x="248" y="15" text-anchor="middle">8</text><text class="viz-label" x="288" y="15" text-anchor="middle">10</text><text class="viz-label" x="328" y="15" text-anchor="middle">12</text>
		<rect class="viz-plot-bg" x="8" y="35" width="340" height="94" rx="4"></rect>
		<text class="viz-callout" x="18" y="53">Baseline</text><text class="viz-label" x="18" y="67">12 cycles</text>
		<rect class="viz-node--focus" x="88" y="45" width="80" height="30" rx="3"></rect><text class="viz-callout" x="128" y="64" text-anchor="middle">load · 4</text>
		<rect class="viz-node--focus" x="168" y="45" width="80" height="30" rx="3"></rect><text class="viz-callout" x="208" y="64" text-anchor="middle">compute · 4</text>
		<rect class="viz-node--focus" x="248" y="45" width="80" height="30" rx="3"></rect><text class="viz-callout" x="288" y="64" text-anchor="middle">store · 4</text>
		<path d="M158 40H178 M238 40H258" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#accelerator-trace-arrow)"></path>
		<rect class="viz-node" x="88" y="88" width="60" height="25" rx="3" style="stroke-dasharray:5 3"></rect><text class="viz-label" x="118" y="104" text-anchor="middle">helper · 3</text><text class="viz-label" x="155" y="104">off-path</text>
		<path class="viz-operating-guide" d="M328 37V121"></path><text class="viz-callout" x="326" y="124" text-anchor="end">end · 12</text>
		<rect class="viz-plot-bg" x="8" y="140" width="340" height="94" rx="4"></rect>
		<text class="viz-callout" x="18" y="158">A · less work</text><text class="viz-label" x="18" y="172">still 12 cycles</text>
		<rect class="viz-node--focus" x="88" y="150" width="80" height="30" rx="3"></rect><text class="viz-callout" x="128" y="169" text-anchor="middle">load · 4</text>
		<rect class="viz-node--focus" x="168" y="150" width="80" height="30" rx="3"></rect><text class="viz-callout" x="208" y="169" text-anchor="middle">compute · 4</text>
		<rect class="viz-node--focus" x="248" y="150" width="80" height="30" rx="3"></rect><text class="viz-callout" x="288" y="169" text-anchor="middle">store · 4</text>
		<path d="M158 145H178 M238 145H258" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#accelerator-trace-arrow)"></path>
		<rect class="viz-node" x="88" y="193" width="60" height="25" rx="3" style="fill:none;stroke-dasharray:5 3"></rect><path d="M93 198L143 213M143 198L93 213" style="stroke:var(--viz-edge);stroke-width:1.5"></path><text class="viz-label" x="155" y="209">helper removed</text>
		<path class="viz-operating-guide" d="M328 142V226"></path><text class="viz-callout" x="326" y="229" text-anchor="end">end · 12</text>
		<rect class="viz-plot-bg" x="8" y="245" width="340" height="127" rx="4"></rect>
		<text class="viz-callout" x="18" y="263">B · shorter chain</text><text class="viz-label" x="18" y="277">10 cycles</text>
		<rect class="viz-node--input" x="88" y="282" width="40" height="30" rx="3" style="stroke-width:2"></rect><text class="viz-callout" x="108" y="301" text-anchor="middle">load · 2</text>
		<rect class="viz-node--focus" x="128" y="282" width="80" height="30" rx="3"></rect><text class="viz-callout" x="168" y="301" text-anchor="middle">compute · 4</text>
		<rect class="viz-node--focus" x="208" y="282" width="80" height="30" rx="3"></rect><text class="viz-callout" x="248" y="301" text-anchor="middle">store · 4</text>
		<path d="M118 277H138 M198 277H218" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#accelerator-trace-arrow)"></path>
		<rect class="viz-node" x="88" y="325" width="60" height="25" rx="3" style="stroke-dasharray:5 3"></rect><text class="viz-label" x="118" y="341" text-anchor="middle">helper · 3</text><text class="viz-label" x="155" y="341">still overlaps</text>
		<path class="viz-operating-guide" d="M288 247V365"></path><text class="viz-callout" x="286" y="368" text-anchor="end">end · 10</text>
		<text class="viz-axis-label" x="208" y="379" text-anchor="middle">solid outline + arrows = dependent critical chain · dashed = off-path</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> compare end markers before counting blocks. Removing the dashed helper lowers total work but leaves the solid 12-cycle load → compute → store chain untouched. Shortening the exposed load advances every dependent operation, so the same remaining work ends at cycle 10. The trace is an original synthetic example; the distinction between elapsed time and summed operation time is checked against the <a href="https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html">NVIDIA Nsight Systems analysis guide</a>, with bottleneck classification informed by the <a href="https://doi.org/10.1145/1498765.1498785">Roofline model</a> and the <a href="https://jax-ml.github.io/scaling-book/profiling/">JAX Scaling Book profiling guide</a>.</figcaption>
</figure>

## What an L4 answer sounds like

> "I would vectorize the loops, unroll them, use all cores, and reduce memory allocations."

Those are common techniques, not a diagnosis. Some increase code volume while leaving the critical path unchanged. Others exceed scratchpad capacity or add gather overhead.

## What an L5 answer adds

An L5 candidate records baseline cycles and correctness, reads the trace, identifies one bottleneck, predicts a result, changes one dimension, and measures. They maintain an experiment ledger and revert regressions.

They reason about trade-offs:

- Larger tiles improve reuse but consume scratchpad.
- SIMD reduces instruction count but may require costly rearrangement or masked lanes.
- VLIW packing improves issue width only when dependencies permit.
- Branch removal can expose parallel work but add computation.
- Prefetching hides latency only when there is independent work to overlap.

## What an L6 answer adds

An L6 candidate protects semantic validity and understands the simulator boundary. They check that an AI agent did not alter tests, core count, instruction costs, data distribution, or machine constraints. A faster invalid program is not an optimization.

They also distinguish simulator wins from hardware wins. A real accelerator has compiler scheduling, cache effects, launch overhead, synchronization, topology, numerical formats, and contention absent from a toy machine. They state what evidence would be needed before transferring the conclusion.

Finally, they build tooling when observation is the bottleneck. A small trace summarizer, dependency visualizer, or scratchpad accounting check can create more speedup than another blind code edit.

## Tells that get you a strong-hire vote

- You verify baseline correctness and cycles first.
- The trace selects the first optimization.
- You predict direction or magnitude before measuring.
- Scratchpad, SIMD occupancy, issue width, and dependency depth are distinct.
- One failed optimization remains in the ledger.
- You revert regressions instead of rationalizing them.
- You audit agent changes for weakened constraints.
- You state why a simulator result may not transfer to hardware.

## Tells that get you down-leveled

- Applying an optimization checklist without reading the trace.
- Equating fewer source lines with fewer cycles.
- Increasing tile size without a memory budget.
- Claiming SIMD speedup while most lanes are idle.
- Changing several mechanisms before measuring.
- Reporting the fastest run without correctness verification.
- Letting an agent modify tests or machine constants.

## Common follow-up

"Instruction count dropped 30%, but cycles did not move. Why?"

The removed instructions were not on the critical path, or another resource became limiting. Inspect dependency stalls, unit saturation, memory waits, and issue packing. Optimization is about elapsed critical-path cycles, not an aggregate count detached from scheduling.

Use the [accelerator lab](/prep/labs/accelerator/) with Anthropic's public challenge and the experiment worksheet.

*Related: [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/), [FlashAttention](/concepts/flashattention/), and [accelerator performance practice](/prep/labs/accelerator/).*
