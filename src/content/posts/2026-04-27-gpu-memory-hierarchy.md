---
title: "GPU memory hierarchy: HBM, SRAM, and roofline reasoning"
description: "Decide whether an accelerator operation is limited by compute or by data movement across HBM, caches, and on-chip memory."
date: "2026-04-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A GPU has small, fast on-chip memory and larger, slower HBM. Large matrix multiplications are often compute-bound. Decode, small matrix multiplications, and many elementwise operations are often memory-bound. The operation's arithmetic intensity determines which limit applies.

Counting only multiply-adds can give the wrong answer. A better first-order model is:

$$
\text{time} \approx \max\!\left(\frac{\text{FLOPs}}{\text{tensor TFLOPs/s}}, \frac{\text{bytes moved}}{\text{HBM TB/s}}\right)
$$

For most LLM ops at typical batch sizes, the second term dominates. This single fact explains:

- Why FlashAttention is faster despite doing the same FLOPs (reduces HBM traffic).
- Why decoding is slow with batch 1 even though the model "fits" (memory-bound).
- Why batching helps decoding so much (amortizes weight reads across many tokens).
- Why low-precision data types help even when the matmul is already fast (less HBM bandwidth).

## The hierarchy

| Tier | Capacity | Bandwidth | Latency | What lives here |
|------|----------|-----------|---------|----------------|
| **Register file** | hundreds of KB per SM | highest on-chip | very low | per-thread values |
| **SRAM (shared memory / L1)** | tens to hundreds of KB per SM | very high | low | tiles for matrix operations |
| **L2 cache** | tens to hundreds of MB per device | below local SRAM | medium | shared across SMs |
| **HBM** | tens to hundreds of GB | below on-chip memory | higher | weights, activations, KV cache |
| **PCIe or accelerator link** | external | below local HBM in many systems | µs scale | host or peer transfer |

These values are illustrative. Capacity, bandwidth, precision, sparsity mode, and power settings vary by accelerator and SKU. Use measured values from the target machine for a deployment plan.

## Arithmetic intensity

For a kernel doing $F$ FLOPs and moving $B$ bytes between HBM and SRAM:

$$
\text{arithmetic intensity} = \frac{F}{B} \quad (\text{FLOPs per byte})
$$

A kernel is **compute-bound** when its intensity exceeds the GPU's peak FLOPs/byte ratio (the "ridge" in a roofline plot). Otherwise it's **memory-bound**.

An accelerator with 989 BF16 TFLOPs/s and 3 TB/s of HBM bandwidth has a ridge near 330 FLOPs/byte. For comparison:

<!-- visual:gpu-roofline-batching-ridge -->
<figure class="learning-figure plot-panel" aria-labelledby="gpu-roofline-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="gpu-roofline-title">When does more compute stop waiting on HBM?</p>
	<svg viewBox="0 0 360 320" role="img" aria-labelledby="gpu-roofline-svg-title gpu-roofline-svg-desc">
		<title id="gpu-roofline-svg-title">Illustrative GPU roofline with decode, batching, a ridge point, and a large matrix multiplication</title>
		<desc id="gpu-roofline-svg-desc">A log-scale roofline for an illustrative accelerator with 989 BF16 teraFLOPs per second and 3 terabytes per second of HBM bandwidth. The sloped bandwidth ceiling reaches the flat compute ceiling at 330 FLOPs per byte. Batch-one decode is near one FLOP per byte. Reusing each weight across a batch of 64 moves decode right to roughly 64 FLOPs per byte but leaves it below the ridge. A 2048 by 2048 square matrix multiplication can reach roughly 1024 FLOPs per byte and lies to the right of the ridge.</desc>
		<defs><marker id="gpu-roofline-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path></marker></defs>
		<rect class="viz-plot-bg" x="48" y="28" width="288" height="222" rx="3"></rect>
		<path class="viz-gridline" d="M48 102H336 M48 176H336 M106 28V250 M164 28V250 M222 28V250 M280 28V250"></path>
		<path class="viz-axis" d="M48 28V250H336"></path>
		<path d="M48 236L288 50H336" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-linejoin:round"></path>
		<path class="viz-operating-guide" d="M288 50V250"></path>
		<text class="viz-callout" x="102" y="170" transform="rotate(-38 102 170)">HBM roof = BW × intensity</text>
		<text class="viz-callout" x="330" y="42" text-anchor="end">compute ceiling = 989 TFLOP/s</text>
		<text class="viz-axis-label" x="98" y="64">MEMORY-BOUND</text>
		<text class="viz-axis-label" x="322" y="115" transform="rotate(90 322 115)">COMPUTE-BOUND</text>
		<circle class="viz-operating-point" cx="48" cy="236" r="5"></circle>
		<text class="viz-callout" x="58" y="220">decode · batch 1</text>
		<text class="viz-label" x="58" y="233">≈ 1 FLOP/byte → ceiling ≈ 3 TFLOP/s</text>
		<path d="M61 229C110 210 168 158 216 112" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#gpu-roofline-arrow)"></path>
		<text class="viz-callout" x="112" y="190">batching reuses each weight</text>
		<circle class="viz-operating-point" cx="222" cy="101" r="5"></circle>
		<text class="viz-callout" x="230" y="120">batch 64</text>
		<text class="viz-label" x="230" y="133">≈ 64 FLOPs/byte</text>
		<polygon class="viz-node--output" points="288,43 295,50 288,57 281,50"></polygon>
		<text class="viz-callout" x="278" y="88" text-anchor="end">ridge ≈ 330</text>
		<rect class="viz-node--output" x="328" y="44" width="10" height="10" rx="1"></rect>
		<text class="viz-callout" x="330" y="68" text-anchor="end">2048² matmul · AI ≈ 1024</text>
		<text class="viz-label" x="45" y="268">1</text><text class="viz-label" x="215" y="268">64</text><text class="viz-label" x="278" y="268">330</text><text class="viz-label" x="319" y="268">1000</text>
		<text class="viz-axis-label" x="192" y="293" text-anchor="middle">arithmetic intensity (FLOPs per HBM byte, log scale)</text>
		<text class="viz-axis-label" x="15" y="210" transform="rotate(-90 15 210)">attainable throughput (log scale)</text>
		<text class="viz-label" x="192" y="311" text-anchor="middle">Upper bounds only; profile the target kernel.</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> start at batch-1 decode: one weight byte supports little work, so the sloped HBM roof is far below peak compute. Batching reuses the loaded weights and moves the operation right, raising its bandwidth-limited ceiling. Only after arithmetic intensity crosses roughly 330 FLOPs/byte does the flat 989-TFLOP/s compute roof become the first-order limit. The log-scale construction and worked points are original; the model is checked against the <a href="https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html">original Roofline report</a>, <a href="https://docs.nersc.gov/tools/performance/roofline/">NERSC's Roofline documentation</a>, and <a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html">NVIDIA's matrix-multiplication guide</a>.</figcaption>
</figure>

- Large square matmul ($n\times n$ by $n\times n$): up to about $n/2$ FLOPs/byte when counting two BF16 inputs and ignoring output traffic. Counting output reads and writes lowers the intensity.
- Attention kernel (without FlashAttention): ~O(d) FLOPs/byte → memory-bound at common d=64–128.
- Single-token decode with BF16 weights: about 1 FLOP per weight byte at batch 1, before other traffic. One multiply-add uses a two-byte weight. This is severely memory-bound.

## Implications for LLMs

- **Training**: large matrix multiplications can be compute-bound, while small batches, elementwise work, and communication can expose other limits.
- **Decode**: often weight-bandwidth-bound at small batch sizes. Batching amortizes weight reads until matrix compute or KV-cache traffic becomes the next limit.
- **KV cache**: bandwidth-dominated read at every decode step. Cache size growth is a serving-throughput issue.

## Common pitfalls

- **Quoting FLOPs as a single-number cost.** Throughput on memory-bound kernels is dictated by bytes, not FLOPs.
- **Assuming peak compute predicts every speedup.** A new accelerator can increase matrix throughput faster than memory bandwidth. Memory-bound work then sees a smaller gain.
- **Ignoring on-chip memory when designing kernels.** FlashAttention tile sizes are constrained by shared memory, registers, and the compiled kernel, not only HBM size.

## Related

- [Accelerator network topology](/concepts/accelerator-network-topology/). Extend the hierarchy across devices and nodes.
- [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/). Convert a model configuration into bytes and FLOPs.
- [Profiling distributed ML workloads](/concepts/profiling-distributed-ml-workloads/). Compare a trace with roofline limits.
