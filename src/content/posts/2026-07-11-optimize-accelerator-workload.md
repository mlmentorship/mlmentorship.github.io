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
