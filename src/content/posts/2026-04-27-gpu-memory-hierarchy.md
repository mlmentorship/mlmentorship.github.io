---
title: "GPU memory hierarchy: HBM, SRAM, and roofline reasoning"
description: "Decide whether an accelerator operation is limited by compute or by data movement across HBM, caches, and on-chip memory."
date: "2026-04-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A kernel must perform arithmetic and move data.
Count both for the same operation before choosing an optimization.

Consider 120 GFLOPs and 12 GB transferred across the measured memory boundary.
Assume hypothetical ceilings of 20 TFLOP/s at the operation's precision and 100 GB/s across that boundary.
Use decimal units: one GB is $10^9$ bytes.
Which could help: twice the compute throughput, or half the traffic?

The compute and bandwidth lower bounds are $120/20000=0.006$ seconds and $12/100=0.12$ seconds.
Even perfect overlap cannot finish below the larger bound:

$$
T\geq\max\!\left(\frac{F}{P},\frac{B}{W}\right).
$$

Here $F$ counts FLOPs, $B$ counts transferred bytes, $P$ is a compute ceiling, and $W$ is a bandwidth ceiling.
Keep units consistent; a multiply-add conventionally counts as two FLOPs.

| Change | Compute bound | Traffic bound | Combined lower bound |
| --- | --- | --- | --- |
| Original | $0.006$ s | $0.12$ s | $0.12$ s |
| Double $P$ | $0.003$ s | $0.12$ s | $0.12$ s |
| Halve $B$ | $0.006$ s | $0.06$ s | $0.06$ s |

Halving traffic improves the lower bound; it does not prove a measured twofold speedup.

## Arithmetic intensity

Arithmetic intensity is work per transferred byte, at a stated memory boundary:

$$
\text{arithmetic intensity} = \frac{F}{B} \quad (\text{FLOPs per byte})
$$

The fixture has intensity $I=10$ FLOPs/byte and ridge $P/W=200$ FLOPs/byte.
Its bandwidth roof is lower than its compute roof:

$$
	ext{throughput}\leq\min(P,WI).
$$

The lower roof identifies a first-order limit, not every possible bottleneck.

<!-- visual:gpu-roofline-batching-ridge -->
<figure class="learning-figure plot-panel" aria-labelledby="gpu-roofline-title">
	<p class="visual-kicker">Same operation, two resource limits</p>
	<p class="visual-title" id="gpu-roofline-title">Count work and traffic at the same boundary; the tighter roof limits the operation</p><div class="visual-argument" data-visual-argument="gpu-roofline-batching-ridge" data-argument-kind="bars" data-argument-composition="bandwidth-ceiling"><svg viewBox="0 0 360 2440" role="img" aria-label="Count work and traffic at the same boundary; the tighter roof limits the operation. This toy needs 120 GFLOPs and 12 GB across the declared memory boundary. Its traffic lower bound dominates compute. Doubling compute leaves that bound unchanged; halving traffic halves it, without proving achieved speedup. Arithmetic intensity and the ridge compare work per byte with hardware balance. Batching may reuse weights but adds other traffic, memory, and waiting. On-chip tiling must fit inputs and accumulators; registers and shared memory have separate limits."><title>Count work and traffic at the same boundary; the tighter roof limits the operation</title><desc>This toy needs 120 GFLOPs and 12 GB across the declared memory boundary. Its traffic lower bound dominates compute. Doubling compute leaves that bound unchanged; halving traffic halves it, without proving achieved speedup. Arithmetic intensity and the ridge compare work per byte with hardware balance. Batching may reuse weights but adds other traffic, memory, and waiting. On-chip tiling must fit inputs and accumulators; registers and shared memory have separate limits.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">The same operation must satisfy</tspan><tspan x="20" dy="18">compute and traffic bounds</tspan></text><text x="20" y="105" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Toy: 120 GFLOPs, 12 GB; 20 TFLOP/s</tspan><tspan x="20" dy="18">and 100 GB/s at the stated precision</tspan><tspan x="20" dy="18">and boundary.</tspan></text><text x="20" y="220" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Roofline: log axes; FLOPs per byte</tspan><tspan x="20" dy="18">across, TFLOP/s up</tspan></text><path d="M60 320L60 620" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M60 620L320 620" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="45" y="625" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="45" dy="0">0.1</tspan></text><text x="45" y="525" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="45" dy="0">1</tspan></text><text x="45" y="425" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="45" dy="0">10</tspan></text><text x="45" y="325" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="45" dy="0">100</tspan></text><text x="60" y="665" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="60" dy="0">1</tspan></text><text x="146.66666666666666" y="665" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="146.66666666666666" dy="0">10</tspan></text><text x="233.33333333333331" y="665" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="233.33333333333331" dy="0">100</tspan></text><text x="320" y="665" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="320" dy="0">1000</tspan></text><path d="M60 389.8970004336019L320 389.8970004336019" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><polyline points="60,620 259.4225996242117,389.8970004336019 320,389.8970004336019" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></polyline><text x="75" y="285" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="75" dy="0">Compute roof: twenty; ridge:</tspan><tspan x="75" dy="18">two hundred</tspan></text><circle cx="146.66666666666666" cy="520" r="5" style="fill:var(--viz-focus-stroke);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><circle cx="172.75593295754504" cy="489.8970004336019" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="20" y="735" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Original intensity ten: roof one.</tspan><tspan x="20" dy="18">Half traffic, intensity twenty: roof</tspan><tspan x="20" dy="18">two.</tspan></text><text x="20" y="855" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Original operation</tspan></text><text x="20" y="908" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Compute lower bound: 0.006 seconds</tspan></text><rect x="20" y="930" width="16" height="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1003" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Traffic lower bound: 0.12 seconds</tspan></text><rect x="20" y="1025" width="320" height="25" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="20" y="1130" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Combined lower bound: 0.12 seconds</tspan></text><text x="20" y="1165" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Double compute only</tspan></text><text x="20" y="1218" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Compute lower bound: 0.003 seconds</tspan></text><rect x="20" y="1240" width="8" height="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1313" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Traffic lower bound: 0.12 seconds</tspan></text><rect x="20" y="1335" width="320" height="25" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="20" y="1440" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Combined lower bound: 0.12 seconds</tspan></text><text x="20" y="1475" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Half memory traffic only</tspan></text><text x="20" y="1528" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Compute lower bound: 0.006 seconds</tspan></text><rect x="20" y="1550" width="16" height="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1623" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Traffic lower bound: 0.06 seconds</tspan></text><rect x="20" y="1645" width="160" height="25" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="20" y="1750" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Combined lower bound: 0.06 seconds</tspan></text><text x="20" y="1875" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Time bars share zero-to-0.12-second</tspan><tspan x="20" dy="18">scale. Lower bounds are not measured</tspan><tspan x="20" dy="18">runtimes.</tspan></text><text x="20" y="1980" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Reuse one BF16 weight across 64</tspan><tspan x="20" dy="18">tokens: about 64 FLOPs per weight</tspan><tspan x="20" dy="18">byte only. Count KV, activation,</tspan><tspan x="20" dy="18">output, memory, and waiting costs</tspan><tspan x="20" dy="18">too.</tspan></text><text x="20" y="2120" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Input tiles 64 KiB plus accumulator</tspan><tspan x="20" dy="18">64 KiB exceed a hypothetical 96 KiB</tspan><tspan x="20" dy="18">local budget.</tspan></text><rect x="20" y="2230" width="160" height="30" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="180" y="2230" width="160" height="30" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><path d="M260 2215L260 2275" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="2340" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Red line: 96 KiB. Separate</tspan><tspan x="20" dy="18">register/shared-memory constraints</tspan><tspan x="20" dy="18">still apply.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 320" role="img" aria-labelledby="gpu-roofline-svg-title gpu-roofline-svg-desc">
		<title id="gpu-roofline-svg-title">Exact log-scale roofline for the hypothetical 20 TFLOP/s and 100 GB/s device</title>
		<desc id="gpu-roofline-svg-desc">The horizontal coordinate is 54 plus 86 times log10 of arithmetic intensity. The vertical coordinate is 236 minus 62 times log10 of throughput divided by 0.1 TFLOP/s. The original operation is at intensity 10 and bandwidth ceiling 1 TFLOP/s. Halving its bytes gives intensity 20 and ceiling 2 TFLOP/s. The ridge is at intensity 200 and throughput 20 TFLOP/s. These are upper bounds, not observed speeds.</desc>
		<defs><marker id="gpu-roofline-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path></marker></defs>
		<rect class="viz-plot-bg" x="54" y="55" width="272" height="195" rx="3"></rect>
		<path class="viz-gridline" d="M54 112H326 M54 174H326 M54 236H326 M140 55V250 M226 55V250 M312 55V250"></path>
		<path class="viz-axis" d="M54 55V250H326"></path>
		<path d="M54 236L251.88858 93.33614H326" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-linejoin:round"></path>
		<path class="viz-operating-guide" d="M251.88858 93.33614V250"></path>
		<text class="viz-callout" x="326" y="42" text-anchor="end">compute ceiling: 20 TFLOP/s</text>
		<text class="viz-callout" x="83" y="129">HBM roof = 0.1 × intensity</text>
		<circle class="viz-operating-point" cx="140" cy="174" r="5"></circle>
		<text class="viz-callout" x="92" y="207">original: 10 → 1 TFLOP/s</text>
		<path d="M147 169L159 160" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#gpu-roofline-arrow)"></path>
		<circle class="viz-operating-point" cx="165.88858" cy="155.33614" r="5"></circle>
		<text class="viz-callout" x="176" y="153">half bytes</text>
		<text class="viz-label" x="176" y="168">20 → 2 TFLOP/s</text>
		<polygon class="viz-node--output" points="251.88858,87.33614 257.88858,93.33614 251.88858,99.33614 245.88858,93.33614"></polygon>
		<text class="viz-callout" x="326" y="78" text-anchor="end">ridge: 200</text>
		<text class="viz-label" x="45" y="240" text-anchor="end">0.1</text>
		<text class="viz-label" x="45" y="178" text-anchor="end">1</text>
		<text class="viz-label" x="45" y="116" text-anchor="end">10</text>
		<text class="viz-label" x="54" y="269" text-anchor="middle">1</text>
		<text class="viz-label" x="140" y="269" text-anchor="middle">10</text>
		<text class="viz-label" x="251.88858" y="269" text-anchor="middle">200</text>
		<text class="viz-label" x="312" y="269" text-anchor="middle">1000</text>
		<text class="viz-axis-label" x="192" y="292" text-anchor="middle">FLOPs per HBM byte (log scale)</text>
		<text class="viz-axis-label" x="15" y="224" transform="rotate(-90 15 224)">TFLOP/s (log scale)</text>
		<text class="viz-label" x="192" y="312" text-anchor="middle">Ceilings only; measure actual kernel time.</text>
	</svg>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> halving bytes doubles arithmetic intensity and its bandwidth ceiling. Both points remain below the compute roof. Doubling compute alone leaves their lower bandwidth roof unchanged.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="gpu-roofline-batching-ridge"><strong>Check:</strong> Why does halving HBM traffic double this ceiling without proving a measured speedup?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="gpu-roofline-batching-ridge">This toy needs 120 GFLOPs and 12 GB across the declared memory boundary. Its traffic lower bound dominates compute. Doubling compute leaves that bound unchanged; halving traffic halves it, without proving achieved speedup. Arithmetic intensity and the ridge compare work per byte with hardware balance. Batching may reuse weights but adds other traffic, memory, and waiting. On-chip tiling must fit inputs and accumulators; registers and shared memory have separate limits. <strong>Does doubling compute change this memory-limited lower bound?</strong></span></figcaption>
</figure>

<p class="diagram-source">Model references: <a href="https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html">original Roofline report</a>, <a href="https://docs.nersc.gov/tools/performance/roofline/">NERSC Roofline documentation</a>, and <a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html">NVIDIA matrix-multiplication guide</a>.</p>

## Change reuse or local capacity

### 1. Reuse weights across a batch

**New task:** small-batch decode repeatedly reads the same BF16 weights.

**Picture and change:** one multiply-add per two-byte weight gives about one FLOP per weight byte.
Reusing each weight across 64 tokens gives about 64, counting weight traffic alone.

**Check:** include activations, outputs and each sequence's KV reads.
Batching also consumes memory and can increase waiting time.

**Use when:** measured weight traffic limits small-batch throughput.

### 2. Reject a tile that cannot stay on chip

**New task:** two BF16 input tiles require $2\times128^2\times2=65536$ bytes.
The kernel also needs a $128^2$ FP32 accumulator: another 65536 bytes.

**Picture and change:** those arrays cannot all reside in a hypothetical 96 KiB local budget.
Choose smaller tiles or a different staging plan.
Registers and shared memory have separate constraints; this combined check is only a necessary fit test.

**Check:** count spills, occupancy, synchronization, output traffic and compiled storage use before claiming reuse.

**Use when:** local capacity invalidates the proposed traffic reduction.

## A faster compute ceiling can leave the limit unchanged

Predicting decode speedup from peak tensor throughput alone ignores bytes and launch costs.
Record shapes, dtype, FLOP convention, memory boundary, traffic, attainable rates and measured runtime.
Compare the same operation and distinguish useful arithmetic from executed or recomputed work.

## Try finding the missing limit

Why might measured runtime exceed both simple lower bounds?

<details>
<summary>Check your answer</summary>

Launch overhead, dependencies, low occupancy, contention, inefficient instructions or uncounted traffic can add limits.
The roofline is not a complete runtime simulator.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## The hierarchy

| Tier | Typical role | Constraint to inspect |
| --- | --- | --- |
| Registers | Per-thread values and accumulators | Register allocation, spills, occupancy |
| Shared memory / L1 | On-chip staging and cached values | Distinct APIs and architecture-specific partitioning |
| L2 | Cache shared across multiprocessors | Actual hit rate and traffic boundary |
| HBM | Weights, activations, KV storage | Capacity and achieved bandwidth |
| PCIe / accelerator links | Host or peer transfers | Link, topology and synchronization |

Capacities and rates vary by device, precision, sparsity mode and power settings.
SRAM is a storage technology, not one universally managed tier.
Measure the target machine and kernel.

## Other intensity calculations

For square BF16 matrix multiplication, $F\approx2n^3$ and two input reads cost $4n^2$ bytes.
Ignoring outputs gives $I\approx n/2$; output reads, writes or repeated loads lower that estimate.
At $n=2048$, this input-only estimate is 1024 FLOPs/byte.

The separate illustrative 989 TFLOP/s, 3 TB/s device has ridge about 330 FLOPs/byte.
It is not the fixture device or a measured kernel result.

## Implications for LLMs

Large matrix multiplications can approach a compute roof; small operations, communication and elementwise kernels can face other limits.
Decode can become limited by weights or KV reads, depending on batch, context and implementation.
FlashAttention reduces materialized attention traffic through tiling; it can also change executed work through recomputation.

## Common pitfalls

Peak figures do not replace profiling.
Include output traffic, spills, cache behavior and the specific boundary at which bytes were measured.

## Related

- [Accelerator network topology](/concepts/accelerator-network-topology/). Extend the hierarchy across devices and nodes.
- [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/). Convert a model configuration into bytes and FLOPs.
- [Profiling distributed ML workloads](/concepts/profiling-distributed-ml-workloads/). Compare a trace with roofline limits.

</div>
