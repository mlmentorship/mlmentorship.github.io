---
title: "FSDP and ZeRO: budget shards and temporary gathers"
description: "Derive persistent bytes at each sharding stage, trace one gathered unit, and test communication and activation-memory changes."
date: "2025-11-14"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["ZeRO-3", "ZeRO stage 3", "fully sharded data parallel", "parameter sharding"]
---

## Summary

Sharding removes replicated stored state; computation can still require a temporary full parameter unit.
Budget both the persistent shards and the buffers live during computation.

**Task:** place one billion parameters across four ranks.
Assume BF16 parameters and gradients, FP32 Adam moments, and a separate FP32 master stored with optimizer shards.
Count decimal GB, not GiB, and initially exclude activations and temporary buffers.
How many persistent bytes remain on each rank?

| Component | Bytes per parameter |
|-----------|----------------------|
| Parameters (BF16) | 2 |
| Gradients (BF16) | 2 |
| Adam first moment $m$ (FP32) | 4 |
| Adam second moment $v$ (FP32) | 4 |
| Separate master weights (FP32, assumed here) | 4 |
| **Total** | **16** |

Without separate masters, this convention totals twelve bytes, not sixteen.
Framework precision and storage choices must be checked rather than inferred from the parameter dtype alone.
The figure traces one fully sharded unit; box widths distinguish ownership from temporary reconstruction, not byte proportions.

<!-- visual:fsdp-wrapped-unit-lifecycle -->
<figure class="learning-figure plot-panel" aria-labelledby="fsdp-lifecycle-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="fsdp-lifecycle-visual-title">Sharded storage still temporarily gathers the full current unit</p><div class="visual-argument" data-visual-argument="fsdp-wrapped-unit-lifecycle" data-argument-kind="ledger" data-argument-composition="transform"><svg viewBox="0 0 360 394" role="img" aria-label="Sharded resting storage still gathers the current unit temporarily. Four ranks each retain one shard at rest. The active wrapped unit gathers for computation, then returns to sharded storage."><title>Sharded resting storage still gathers the current unit temporarily</title><desc>Four ranks each retain one shard at rest. The active wrapped unit gathers for computation, then returns to sharded storage.</desc><circle cx="23" cy="24" r="10" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="23" y="28" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">1</tspan></text><text x="44" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">At rest on each rank</tspan></text><g data-proof-token="1/4" data-proof-row="0" data-proof-position="0" data-proof-box="44,60,68,48"><rect x="44" y="60" width="68" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="78" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="78" dy="0">1/4</tspan></text></g><g data-proof-token="1/4" data-proof-row="0" data-proof-position="1" data-proof-box="118,60,68,48"><rect x="118" y="60" width="68" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="152" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="152" dy="0">1/4</tspan></text></g><g data-proof-token="1/4" data-proof-row="0" data-proof-position="2" data-proof-box="192,60,68,48"><rect x="192" y="60" width="68" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="226" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="226" dy="0">1/4</tspan></text></g><g data-proof-token="1/4" data-proof-row="0" data-proof-position="3" data-proof-box="266,60,68,48"><rect x="266" y="60" width="68" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="300" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="300" dy="0">1/4</tspan></text></g><path d="M23 41L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="140" r="10" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="23" y="144" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">2</tspan></text><text x="44" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Active unit temporarily on each</tspan><tspan x="44" dy="18">rank</tspan></text><g data-proof-token="full" data-proof-row="1" data-proof-position="0" data-proof-box="44,194,68,48"><rect x="44" y="194" width="68" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="78" y="221" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="78" dy="0">full</tspan></text></g><g data-proof-token="full" data-proof-row="1" data-proof-position="1" data-proof-box="118,194,68,48"><rect x="118" y="194" width="68" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="152" y="221" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="152" dy="0">full</tspan></text></g><g data-proof-token="full" data-proof-row="1" data-proof-position="2" data-proof-box="192,194,68,48"><rect x="192" y="194" width="68" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="226" y="221" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="226" dy="0">full</tspan></text></g><g data-proof-token="full" data-proof-row="1" data-proof-position="3" data-proof-box="266,194,68,48"><rect x="266" y="194" width="68" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="300" y="221" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="300" dy="0">full</tspan></text></g><path d="M23 157L23 252" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 246L23 252" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 246L23 252" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="274" r="10" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="23" y="278" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">3</tspan></text><text x="44" y="278" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">After unit computation</tspan></text><g data-proof-token="1/4" data-proof-row="2" data-proof-position="0" data-proof-box="44,310,68,48"><rect x="44" y="310" width="68" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="78" y="339.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="78" dy="0">1/4</tspan></text></g><g data-proof-token="1/4" data-proof-row="2" data-proof-position="1" data-proof-box="118,310,68,48"><rect x="118" y="310" width="68" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="152" y="339.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="152" dy="0">1/4</tspan></text></g><g data-proof-token="1/4" data-proof-row="2" data-proof-position="2" data-proof-box="192,310,68,48"><rect x="192" y="310" width="68" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="226" y="339.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="226" dy="0">1/4</tspan></text></g><g data-proof-token="1/4" data-proof-row="2" data-proof-position="3" data-proof-box="266,310,68,48"><rect x="266" y="310" width="68" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="300" y="339.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="300" dy="0">1/4</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 520" role="img" aria-labelledby="fsdp-lifecycle-svg-title fsdp-lifecycle-svg-desc">
		<title id="fsdp-lifecycle-svg-title">Lifecycle of one wrapped unit on one rank under FSDP full sharding</title>
		<desc id="fsdp-lifecycle-svg-desc">At rest, rank i stores only parameter shard P i, a slot for gradient shard G i, and optimizer shard O i. Before forward, an all-gather reconstructs the current wrapped unit's full parameters P zero through P three temporarily. Forward compute runs, then the full parameters are freed while P i remains. Before backward, another all-gather reconstructs the full unit. Backward compute produces gradients, and reduce-scatter leaves gradient shard G i on this rank. The local optimizer shard O i uses G i to update P i. Activations and communication buffers are additional memory and are not shown.</desc>
		<text class="viz-axis-label" x="20" y="22">RANK i · ONE WRAPPED UNIT · FULL_SHARD</text>
		<text class="viz-callout" x="20" y="48">Persistent rank-local state</text>
		<rect class="viz-node viz-node--input" x="20" y="60" width="96" height="52" rx="4"></rect>
		<text class="viz-node-value" x="68" y="81">PARAMETER SHARD</text>
		<text class="viz-node-label" x="68" y="101">Pᵢ</text>
		<rect class="viz-node" x="132" y="60" width="96" height="52" rx="4"></rect>
		<text class="viz-node-value" x="180" y="81">GRADIENT SLOT</text>
		<text class="viz-node-label" x="180" y="101">Gᵢ</text>
		<rect class="viz-node viz-node--output" x="244" y="60" width="96" height="52" rx="4"></rect>
		<text class="viz-node-value" x="292" y="81">OPTIMIZER SHARD</text>
		<text class="viz-node-label" x="292" y="101">Oᵢ</text>
		<path class="viz-axis" d="M180 112 V142"></path><path class="viz-arrow-forward" d="M180 148 l-5 -9 h10 Z"></path>
		<text class="viz-edge-label" x="180" y="132">all-gather parameter shards</text>
		<text class="viz-callout" x="20" y="166">Temporary forward window</text>
		<rect class="viz-node viz-node--focus" x="20" y="178" width="320" height="58" rx="4"></rect>
		<text class="viz-node-value" x="180" y="199">FULL CURRENT UNIT ON THIS RANK</text>
		<text class="viz-node-label" x="180" y="222">P₀ | P₁ | P₂ | P₃ → forward compute</text>
		<path class="viz-axis" d="M180 236 V266"></path><path class="viz-arrow-forward" d="M180 272 l-5 -9 h10 Z"></path>
		<text class="viz-edge-label" x="180" y="256">reshard · free full parameters</text>
		<rect class="viz-node viz-node--input" x="100" y="280" width="160" height="42" rx="4"></rect>
		<text class="viz-node-value" x="180" y="297">BETWEEN COMPUTE WINDOWS</text>
		<text class="viz-node-label" x="180" y="315">keep Pᵢ, not full P</text>
		<path class="viz-axis" d="M180 322 V352"></path><path class="viz-arrow-forward" d="M180 358 l-5 -9 h10 Z"></path>
		<text class="viz-edge-label" x="180" y="342">all-gather parameter shards again</text>
		<text class="viz-callout" x="20" y="376">Temporary backward window</text>
		<rect class="viz-node viz-node--focus" x="20" y="388" width="320" height="52" rx="4"></rect>
		<text class="viz-node-value" x="180" y="408">FULL CURRENT UNIT ON THIS RANK</text>
		<text class="viz-node-label" x="180" y="430">backward compute → full gradients</text>
		<path class="viz-axis" d="M180 440 V466"></path><path class="viz-arrow-forward" d="M180 472 l-5 -9 h10 Z"></path>
		<text class="viz-edge-label" x="180" y="458">reduce-scatter gradients · free full unit</text>
		<rect class="viz-node viz-node--output" x="20" y="480" width="320" height="28" rx="4"></rect>
		<text class="viz-callout" x="180" y="499" text-anchor="middle">local step: Oᵢ + Gᵢ updates Pᵢ</text>
	</svg>
	<div class="visual-derivation-caption">narrow boxes show what rank <var>i</var> owns across the step. Wide boxes show the temporary full unit gathered for compute. This policy frees parameters after forward, so backward needs another gather. Reduce-scatter leaves <var>Gᵢ</var>; the local optimizer updates <var>Pᵢ</var>. Activations and communication buffers add to the peak. Original schematic checked against the <a href="https://arxiv.org/abs/1910.02054">ZeRO paper</a>, <a href="https://docs.pytorch.org/docs/2.13/fsdp.html">PyTorch FSDP documentation</a>, and <a href="https://deepspeed.readthedocs.io/en/stable/zero3.html">DeepSpeed ZeRO documentation</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="fsdp-wrapped-unit-lifecycle"><strong>Check:</strong> Why does parameter sharding not mean a rank never temporarily holds the full current unit?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="fsdp-wrapped-unit-lifecycle">Four ranks each retain one shard at rest. The active wrapped unit gathers for computation, then returns to sharded storage.</span></figcaption>
</figure>

## The three stages (ZeRO-1/2/3)

Let $P$ be parameter count and $N$ the data-parallel group size.
Optimizer storage here includes eight moment bytes and four master-weight bytes per parameter.

### Stage 1: shard optimizer state

Parameters and gradients stay replicated; optimizer storage divides across ranks:
$M_1=P(4+12/N)$ bytes.
For the fixture, $M_1=4+12/4=7$ GB, versus the replicated $16$ GB.
Each rank updates its owned slice, then shares updated parameters as required by the implementation.

### Stage 2: shard optimizer state + gradients

Only parameters stay replicated: $M_2=P(2+14/N)=5.5$ GB here.
Reduced gradient shards support local optimizer updates.
The large-$N$ reduction tends to eightfold under this sixteen-byte convention; at four ranks it is only $16/5.5\approx2.91$.

### Stage 3 (FSDP): shard optimizer state + gradients + parameters

Ideal persistent storage becomes $M_3=16P/N=4$ GB.
All three formulas return sixteen GB at $N=1$.
FSDP and ZeRO-3 implement related full-sharding approaches, not identical APIs or peak-memory policies.

The lifecycle is gather, forward, free, gather again, backward, reduce-scatter, and local update.
Prefetch and retained units can overlap those windows and increase live memory.

### Trade a gather for retention

**New task:** reduce communication when memory has headroom.
Suppose a wrapped unit contains $250$ million BF16 parameters: its full gathered buffer is $0.5$ GB.
Assume the four-GB persistent shard remains allocated alongside two-GB activations and $0.5$ GB of other buffers.
Conservatively counting a separate full gather gives $4+2+0.5+0.5=7$ GB.

**Check:** retaining this unit through backward avoids its second gather but keeps that buffer live.
An ideal four-rank ring gather sends and receives $3/4\times0.5=0.375$ GB per rank.
Retaining four equal units instead raises this ledger to $8.5$ GB, exceeding an eight-GB device.
**Use when:** measured communication warrants a policy whose complete live-buffer set fits.

### Change the dominant memory

**New task:** activations grow to six GB while local batch and sequence length stay fixed across rank comparisons.
At four ranks, the same ledger becomes $4+6+0.5+0.5=11$ GB.
At eight ranks, persistent state halves, but $2+6+0.5+0.5=9$ GB still exceeds eight GB.

**Check:** a hypothetical checkpointing change reducing activations to two GB gives five GB at eight ranks, with recomputation cost.
**Use when:** activations dominate; state sharding alone does not divide that term.

### Failure and transfer

A giant wrapping unit can exceed capacity even when persistent shards are small.
**Reconstruct:** why can smaller units be slower?

<details><summary>Answer</summary>

Smaller gathers reduce individual buffers but increase collective launches and can weaken efficient overlap.
Choose from measured peak liveness and step time, not persistent bytes alone.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Tradeoffs

- Wrapping determines gather size, launch count, and parameter lifetime; prefetch can make multiple full units overlap.
- Gather retention trades traffic for live memory; allocator reuse and storage aliasing change the actual peak.
- Shard counts can be uneven, especially with small tensors or padding.
- Hybrid groups shard within a selected group and replicate across groups; use that sharding degree, not total GPU count.
- Checkpoint formats and reshard-on-load policies must preserve parameter identity, optimizer state, and update count.

## When to use what

| Constraint | Strategy |
|-----------|----------|
| Replicated state fits across the target data-parallel workers | Consider DDP without state sharding |
| Persistent optimizer state is the first limit | ZeRO-1 |
| Optimizer state and gradients are the limit | ZeRO-2 |
| Stored parameters are also the limit | FSDP / ZeRO-3 |
| One gathered layer is too large | Revisit wrapping or consider tensor parallelism |
| Depth or topology needs another split | Consider pipeline parallelism |

## Common pitfalls

FSDP does not inherently partition every activation dimension.
Use [activation checkpointing](/concepts/activation-checkpointing/) or an appropriate sequence/context layout for the actual dominant term.
Tensor parallelism partitions a layer's computation; it has different layout and collective contracts.
The numerical ledgers above are hypothetical accounting examples, not measured allocator peaks or performance claims.

## Related

- [Strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/). Choose the lightest sharding stage that fits.
- [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/). Connect layouts with all-gather and reduce-scatter.
- [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/). Build the memory budget first.

</div>
