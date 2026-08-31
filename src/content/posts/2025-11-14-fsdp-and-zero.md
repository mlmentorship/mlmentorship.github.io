---
title: "FSDP and ZeRO: sharding optimizer state, gradients, and parameters"
description: "How modern training scales beyond a single GPU's memory by partitioning the optimizer state, gradients, and parameters across the data-parallel group."
date: "2025-11-14"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["ZeRO-3", "ZeRO stage 3", "fully sharded data parallel", "parameter sharding"]
---

## Summary

ZeRO stages progressively partition optimizer state, gradients, and model parameters across data-parallel GPUs. Fully Sharded Data Parallel (FSDP) partitions all three persistent state types. Earlier ZeRO stages leave some state replicated.

Training memory has four big consumers:

| Component | Bytes per parameter (BF16 + FP32 master + Adam) |
|-----------|----------------------|
| Parameters (BF16) | 2 |
| Gradients (BF16) | 2 |
| Adam first moment $m$ (FP32) | 4 |
| Adam second moment $v$ (FP32) | 4 |
| Master weights (FP32, optional) | 4 |
| **Total** | **~12–16 bytes/param** |

A 7B-parameter model therefore needs about 84–112 GB of persistent training state before activations and temporary buffers. The lower value omits separate FP32 master weights. The exact value depends on optimizer and framework storage. Ideal full sharding spreads persistent state across $N$ GPUs, but each GPU also needs transient gathered parameters, communication buffers, and activations.

<!-- visual:fsdp-wrapped-unit-lifecycle -->
<figure class="learning-figure plot-panel" aria-labelledby="fsdp-lifecycle-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="fsdp-lifecycle-visual-title">Separate persistent shards from the temporary full unit needed for compute.</p>
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
	<figcaption><strong>Read it this way:</strong> read the narrow shard boxes as what rank <var>i</var> owns across the step. The wide boxes are temporary peaks: full parameters for only the active wrapped unit are all-gathered for forward and again for backward, then freed. Reduce-scatter returns only <var>Gᵢ</var>, so the local optimizer shard can update <var>Pᵢ</var>. Activations and communication buffers still add to peak memory. Original schematic checked against the <a href="https://arxiv.org/abs/1910.02054">ZeRO paper</a>, <a href="https://docs.pytorch.org/docs/2.13/fsdp.html">PyTorch FSDP documentation</a>, and <a href="https://deepspeed.readthedocs.io/en/stable/zero3.html">DeepSpeed ZeRO documentation</a>.</figcaption>
</figure>

ZeRO ([Rajbhandari et al., 2019](https://arxiv.org/abs/1910.02054), DeepSpeed) and PyTorch FSDP implement this idea. They are the standard for any training run that doesn't fit in a single GPU's memory and doesn't need full tensor or pipeline parallelism.

## The three stages (ZeRO-1/2/3)

### Stage 1: shard optimizer state
Each GPU holds the full parameters and gradients but only $1/N$ of the Adam moments. After the backward pass, each GPU updates its slice and then all-gathers updated parameters.
Memory reduction: up to about 4× for large groups, depending on master-weight storage.

### Stage 2: shard optimizer state + gradients
Same as Stage 1 plus gradients are reduced-scattered (each GPU keeps its slice) instead of all-reduced.
Memory reduction: up to about 8× for large groups under the 16-byte assumption.

### Stage 3 (FSDP): shard optimizer state + gradients + parameters
Each GPU holds only its persistent slice of the parameters. It all-gathers one wrapped unit before computing that unit. If the full parameters are freed after forward, they must be gathered again for backward. Keeping them through backward saves that second gather but raises peak memory.
Memory reduction: nearly $N$× for persistent model state. Peak memory also includes at least one gathered unit and temporary buffers.

PyTorch FSDP and DeepSpeed ZeRO-3 are common implementations of full parameter sharding.

## Tradeoffs

- **Memory vs. communication**: each stage trades more communication for less memory.
- **Sharding granularity**: FSDP can wrap individual layers ("auto-wrap policy") so all-gathers cover only one layer's parameters at a time, capping peak unsharded memory.
- **Mixing with tensor parallelism**: FSDP shards across the data-parallel dimension. Very large runs often combine it with tensor parallelism on the fastest links. The exact placement follows the cluster topology.

## When to use what

| Constraint | Strategy |
|-----------|----------|
| Fits on 1 GPU | DDP (no sharding) |
| Persistent optimizer state is the first limit | ZeRO-1 |
| Optimizer state and gradients are the limit | ZeRO-2 |
| Stored parameters are also the limit | FSDP / ZeRO-3 |
| One gathered layer is too large | Add tensor parallelism |
| Depth or topology needs another split | Consider pipeline parallelism |

## Common pitfalls

- **FSDP alone does not shard activation dimensions.** Each GPU still holds activations for its data-parallel slice. Use [activation checkpointing](/concepts/activation-checkpointing/) or sequence/context parallelism when those activations dominate.
- **All-gather overhead at small layer size.** Wrapping every linear layer separately can dominate runtime; wrap at transformer-block granularity instead.
- **Confusing sharding with tensor parallelism.** Sharding (FSDP) splits state across data-parallel ranks and reconstructs it for compute. Tensor parallelism splits the *compute* of a single layer; the math is different.

## Related

- [Strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/). Choose the lightest sharding stage that fits.
- [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/). Connect layouts with all-gather and reduce-scatter.
- [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/). Build the memory budget first.
