---
title: "Tensor parallelism"
description: "Split a single matrix multiplication across multiple GPUs. The way to fit one transformer layer that doesn't fit on a single device."
date: "2025-12-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Tensor parallelism** (TP) splits the *computation* of a single layer (typically a matmul) across multiple GPUs by sharding the weight matrix along one of its dimensions. Each GPU computes its slice, and an all-reduce or all-gather aggregates the result before the next layer.

For very large models (70B+, 405B, MoE-1T), a single transformer layer's weights and activations don't fit on a single GPU even with FSDP. TP shards individual layers. Required for frontier-scale training and inference. Combined with pipeline parallelism and FSDP, it forms **3D parallelism** used by modern training stacks.

## How a transformer layer is sharded

The standard sharding from Megatron-LM [(Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053):

### FFN (two matmuls + activation)

```
y = GeLU(x @ W_1) @ W_2
```

<!-- visual:tensor-parallel-ffn-handoff -->
<figure class="learning-figure" aria-labelledby="tensor-parallel-ffn-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="tensor-parallel-ffn-title">Follow two matching channel shards through an FFN and identify the first point that requires communication.</p>
	<div class="visual-grid--two" role="group" aria-label="Column-parallel first projection followed by row-parallel second projection">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 245" role="img" aria-labelledby="tp-column-title tp-column-desc">
				<title id="tp-column-title">The column-parallel first projection creates independent channel shards</title>
				<desc id="tp-column-desc">The replicated input x is multiplied independently by the left and right column shards of W one. GPU zero produces activated channel shard h zero, and GPU one produces activated channel shard h one. GeLU is elementwise, so no collective is needed between the first projection and activation.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="212" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">1 · COLUMN-SPLIT W₁ · OUTPUT CHANNELS SPLIT</text>
				<rect class="viz-node viz-node--input" x="113" y="38" width="74" height="34" rx="4"></rect>
				<text class="viz-node-label" x="150" y="59">x replicated</text>
				<path d="M136 72L87 98M164 72L213 98" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<rect class="viz-node" x="23" y="99" width="124" height="55" rx="4"></rect>
				<rect class="viz-node" x="153" y="99" width="124" height="55" rx="4" style="stroke-dasharray:5 3"></rect>
				<text class="viz-node-value" x="85" y="119">GPU 0 · LEFT COLUMNS</text>
				<text class="viz-node-label" x="85" y="140">x W₁ᵃ</text>
				<text class="viz-node-value" x="215" y="119">GPU 1 · RIGHT COLUMNS</text>
				<text class="viz-node-label" x="215" y="140">x W₁ᵇ</text>
				<path d="M85 154V176M215 154V176" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<rect class="viz-node viz-node--output" x="38" y="177" width="94" height="36" rx="4"></rect>
				<rect class="viz-node viz-node--output" x="168" y="177" width="94" height="36" rx="4" style="stroke-dasharray:5 3"></rect>
				<text class="viz-node-label" x="85" y="199">hᵃ = GeLU(·)</text>
				<text class="viz-node-label" x="215" y="199">hᵇ = GeLU(·)</text>
				<text class="viz-axis-label" x="150" y="229" text-anchor="middle">NO COLLECTIVE · CHANNEL SHARDS STAY SEPARATE</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 245" role="img" aria-labelledby="tp-row-title tp-row-desc">
				<title id="tp-row-title">The row-parallel second projection creates partial sums that must be reduced</title>
				<desc id="tp-row-desc">Channel shard h zero multiplies the matching top row shard W two a on GPU zero. Channel shard h one multiplies the matching bottom row shard W two b on GPU one. Both products have the full output shape but each sums over only half the hidden channels. An all-reduce adds the partial outputs p zero and p one to form y.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="212" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">2 · ROW-SPLIT W₂ · PARTIAL OUTPUTS SUM</text>
				<rect class="viz-node viz-node--input" x="22" y="39" width="125" height="46" rx="4"></rect>
				<rect class="viz-node viz-node--input" x="153" y="39" width="125" height="46" rx="4" style="stroke-dasharray:5 3"></rect>
				<text class="viz-node-value" x="84.5" y="57">GPU 0 · MATCH SHARD a</text>
				<text class="viz-node-label" x="84.5" y="75">pᵃ = hᵃ W₂ᵃ</text>
				<text class="viz-node-value" x="215.5" y="57">GPU 1 · MATCH SHARD b</text>
				<text class="viz-node-label" x="215.5" y="75">pᵇ = hᵇ W₂ᵇ</text>
				<path d="M84 85V112H132M216 85V112H168" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<circle cx="150" cy="112" r="18" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
				<text class="viz-node-label" x="150" y="118">+</text>
				<text class="viz-axis-label" x="150" y="147" text-anchor="middle">ALL-REDUCE ACROSS THE TP GROUP</text>
				<path d="M150 130V162" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<rect class="viz-node viz-node--output" x="69" y="163" width="162" height="43" rx="4"></rect>
				<text class="viz-node-value" x="150" y="181">REPLICATED OUTPUT</text>
				<text class="viz-node-label" x="150" y="198">y = pᵃ + pᵇ</text>
				<text class="viz-axis-label" x="150" y="229" text-anchor="middle">SAME OUTPUT SHAPE · DISJOINT SUM TERMS</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> split <em>W₁</em> by output channels, so each GPU can apply GeLU to its own <em>h</em> shard. Feed each shard directly into the matching rows of <em>W₂</em>. Those second products are partial sums over the hidden dimension, so add them with one all-reduce to recover <em>y</em>.</figcaption>
</figure>

- $W_1$ split column-wise: each GPU holds $W_1[:, c_i:c_{i+1}]$. Produces a partial output for its slice of channels. No communication needed up to the GeLU (elementwise).
- $W_2$ split row-wise: each GPU holds $W_2[r_i:r_{i+1}, :]$. Multiplies its slice. Output is summed across GPUs via all-reduce.

Two matmuls with one all-reduce per FFN block.

### Attention

Split heads across GPUs: each GPU computes its subset of attention heads. Output projection $W_O$ is split row-wise, requiring an all-reduce at the end.

Two matmuls (heads, output projection) with one all-reduce per attention block.

## Communication cost

In the common Megatron layout, a forward pass uses one activation reduction after the attention output projection and one after the second feed-forward projection. The backward pass has matching communication for the input gradients of the column-parallel projections.

The message size follows the activation shape, not the parameter count. If an activation tensor contains $A$ bytes and the tensor-parallel degree is $N$, a ring all-reduce moves about:

$$
2A\frac{N-1}{N}
$$

bytes per rank. Sequence-parallel implementations often replace an all-reduce with a reduce-scatter and a later all-gather so the intermediate activation stays split. The exact collective count depends on the layout and framework.

TP communicates every layer, so it needs high effective bandwidth and low latency. It is commonly kept inside a fast accelerator domain. It can cross nodes when the network, message sizes, and local batch provide enough communication efficiency. The decision should come from a cost estimate and a scaling trace, not a fixed node boundary.

## Sequence parallelism

A complement to TP that shards the **sequence dimension** for operations not parallelized by TP, such as LayerNorm, dropout, and residual work. It can reduce those activation tensors by about the tensor-parallel degree. It is not free: it usually changes all-reduce operations into paired reduce-scatter and all-gather operations. This can keep similar communication volume while reducing peak memory.

## TP vs. data parallelism vs. pipeline parallelism

| Sharding axis | Memory savings | Communication |
|---------------|---------------|---------------|
| **DDP / FSDP** (data) | Each GPU sees a different mini-batch | Gradient all-reduce / all-gather |
| **TP** (tensor) | Each GPU shards layer weights and activations | Per-layer all-reduce |
| **PP** (pipeline) | Each GPU holds different layers | Activation send between adjacent stages |
| **Sequence** (within TP) | Reduces activation memory in TP | Reduce-scatter and all-gather layout changes |

**3D parallelism**: combine DP + TP + PP for very large models. Typical config: TP within a node, PP across small groups of nodes, DP across remaining nodes.

## When to use TP

- **Layer too large to fit on single GPU**: even with FSDP all-gather, the unsharded layer must fit. TP keeps the layer sharded throughout.
- **Inference**: TP is a common way to serve models that need several accelerators; major serving runtimes support it.
- **Throughput optimization within a node**: TP with NVLink can be faster than data parallelism for small batch sizes.

## When NOT to use TP

- **When the required links are too slow**: estimate exposed collective time for the real message sizes before extending the group.
- **Small models that fit on one GPU**: pure DP / FSDP is simpler.
- **Pipeline-friendly architectures**: PP can be cheaper communication-wise across slow interconnects.

## Common pitfalls

- **Using TP across a slow interconnect.** Frequent activation collectives can dominate. Keep the group on the fastest useful links unless measurements support a wider group.
- **Assuming TP solves every memory limit.** TP shards layers along selected axes. Add optimizer or full-state sharding only when the remaining state requires it.
- **Sharding embedding tables incorrectly.** The vocab embedding is large ($V \times d$); shard it carefully (Megatron has its own embedding sharding).
- **Communication count math.** Each TP block adds all-reduces; for narrow models / small batches, communication can dominate compute.
- **Tooling ambiguity.** "Tensor parallel size = 8" with mismatched DP / PP can give surprising aggregate batch sizes.

## Related

- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Orthogonal sharding.
- [Pipeline parallelism](/concepts/pipeline-parallelism/). Alternative cross-device split.
- [All-reduce and collectives](/concepts/all-reduce-and-collectives/). The underlying primitives.
- [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/). Derive collectives from tensor layouts.
- [Accelerator network topology](/concepts/accelerator-network-topology/). Place the tensor-parallel group.
