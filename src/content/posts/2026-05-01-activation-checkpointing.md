---
title: "Activation checkpointing"
description: "Trade compute for memory: drop activations during the forward pass and recompute them during the backward pass. The cheapest way to fit a larger model on the same GPU."
date: "2026-05-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Activation checkpointing (also called gradient checkpointing) saves only a subset of activations during the forward pass and recomputes the rest from those saved checkpoints during the backward pass. Memory drops at the cost of one extra forward pass per checkpoint segment.

Backprop needs every layer's input activation to compute that layer's parameter gradient. For a deep model the activations dominate training memory. Often more than parameters and optimizer state combined. A 7B-parameter transformer with 32 layers, batch 1, sequence 4096 stores tens of GB of activations.

Checkpointing recovers this memory by repeating part of the forward work. Recomputing the full forward graph adds about one forward pass of FLOPs. Wall-time cost depends on kernels, memory traffic, and how much of the graph is recomputed.

## The mechanism

Partition a chain of $L$ layers into segments of $K$ consecutive layers. During forward:

1. Run the segment.
2. Save **only** its input (the checkpoint).
3. Discard intermediate activations.

During backward:

1. Recompute the segment's forward pass starting from the saved input.
2. Compute gradients normally for that segment.
3. Discard the recomputed activations.

For a transformer, the natural segment is one transformer block. PyTorch provides `torch.utils.checkpoint.checkpoint(...)` and `checkpoint_sequential(...)`; modern training stacks expose this as a single flag (e.g., `gradient_checkpointing=True` in HuggingFace `Trainer`).

## Cost model

- **Memory across a simple chain**: if each segment has $K$ layers, peak saved boundary state scales like $O(L/K)$ and peak temporary recomputation state like $O(K)$. The total $O(L/K + K)$ is minimized near $K=\sqrt{L}$.
- **Memory inside a transformer block**: checkpointing every block still stores block boundaries, but discards the larger internal matrix and attention intermediates. The reduction is a workload-dependent constant factor, not $L$×.
- **Compute**: full rematerialization adds roughly one forward pass to a training step. Since a forward and backward step is often estimated at three forward-pass units, this is about 33% more FLOPs. Wall-time overhead can differ.

<!-- visual:checkpoint-boundary-recompute-balance -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="checkpoint-balance-title">
	<p class="visual-kicker">Memory at one backward step</p>
	<p class="visual-title" id="checkpoint-balance-title">Saved boundaries and one rebuilt segment coexist in memory.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 700 330" role="img" aria-labelledby="checkpoint-balance-svg-title checkpoint-balance-svg-desc">
			<title id="checkpoint-balance-svg-title">Activation checkpoint memory for a sixteen-layer chain split into four-layer segments</title>
			<desc id="checkpoint-balance-svg-desc">Sixteen numbered layer boxes are divided into four segments of four layers. Diamond checkpoints mark the four saved segment inputs. During backward, a dashed enclosure marks the four activations temporarily rebuilt for layers nine through twelve. Below, four checkpoint diamonds plus four rebuilt activation squares total eight illustrative activation-sized units. The diagram concludes that balancing L divided by K against K gives a segment length near the square root of L.</desc>
			<text class="viz-axis-label" x="22" y="24">EXAMPLE: L = 16 layers, K = 4 layers per segment</text>
			<text class="viz-label" x="22" y="47">Forward keeps only each segment input (diamond).</text>
			<path class="viz-axis" d="M42 92 H652"></path>
			<g aria-label="Sixteen layers in four segments">
				<rect class="viz-node" x="42" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="55.5" y="96">1</text>
				<rect class="viz-node" x="80" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="93.5" y="96">2</text>
				<rect class="viz-node" x="118" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="131.5" y="96">3</text>
				<rect class="viz-node" x="156" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="169.5" y="96">4</text>
				<rect class="viz-node" x="194" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="207.5" y="96">5</text>
				<rect class="viz-node" x="232" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="245.5" y="96">6</text>
				<rect class="viz-node" x="270" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="283.5" y="96">7</text>
				<rect class="viz-node" x="308" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="321.5" y="96">8</text>
				<rect class="viz-node viz-node--focus" x="346" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="359.5" y="96">9</text>
				<rect class="viz-node viz-node--focus" x="384" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="397.5" y="96">10</text>
				<rect class="viz-node viz-node--focus" x="422" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="435.5" y="96">11</text>
				<rect class="viz-node viz-node--focus" x="460" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="473.5" y="96">12</text>
				<rect class="viz-node" x="498" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="511.5" y="96">13</text>
				<rect class="viz-node" x="536" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="549.5" y="96">14</text>
				<rect class="viz-node" x="574" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="587.5" y="96">15</text>
				<rect class="viz-node" x="612" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="625.5" y="96">16</text>
			</g>
			<g aria-label="Four saved checkpoint boundaries">
				<path class="viz-node--output" d="M35 92 l7 -7 l7 7 l-7 7 Z"></path>
				<path class="viz-node--output" d="M187 92 l7 -7 l7 7 l-7 7 Z"></path>
				<path class="viz-node--output" d="M339 92 l7 -7 l7 7 l-7 7 Z"></path>
				<path class="viz-node--output" d="M491 92 l7 -7 l7 7 l-7 7 Z"></path>
			</g>
			<path class="viz-operating-guide" d="M340 64 H493 V120 H340 Z"></path>
			<text class="viz-callout" x="416.5" y="139" text-anchor="middle">one segment rebuilt during backward</text>
			<text class="viz-axis-label" x="22" y="178">PEAK ACTIVATION MEMORY IN THIS SIMPLE CHAIN</text>
			<text class="viz-label" x="22" y="201">persistent boundaries</text>
			<g aria-label="Four persistent boundary checkpoints">
				<path class="viz-node--output" d="M55 231 l10 -10 l10 10 l-10 10 Z"></path>
				<path class="viz-node--output" d="M87 231 l10 -10 l10 10 l-10 10 Z"></path>
				<path class="viz-node--output" d="M119 231 l10 -10 l10 10 l-10 10 Z"></path>
				<path class="viz-node--output" d="M151 231 l10 -10 l10 10 l-10 10 Z"></path>
			</g>
			<text class="viz-callout" x="183" y="235">L/K = 4</text>
			<text class="viz-callout" x="255" y="235">+</text>
			<text class="viz-label" x="292" y="201">temporary rebuilt activations</text>
			<g aria-label="Four temporary recomputed activations">
				<rect class="viz-node viz-node--focus" x="304" y="218" width="26" height="26" rx="3"></rect>
				<rect class="viz-node viz-node--focus" x="338" y="218" width="26" height="26" rx="3"></rect>
				<rect class="viz-node viz-node--focus" x="372" y="218" width="26" height="26" rx="3"></rect>
				<rect class="viz-node viz-node--focus" x="406" y="218" width="26" height="26" rx="3"></rect>
			</g>
			<text class="viz-callout" x="452" y="235">K = 4</text>
			<text class="viz-callout" x="515" y="235">= 8 units</text>
			<path class="viz-baseline" d="M22 266 H678"></path>
			<text class="viz-label" x="22" y="291">smaller K → more saved boundaries</text>
			<text class="viz-label" x="678" y="291" text-anchor="end">larger K → larger rebuilt window</text>
			<text class="viz-callout" x="350" y="317" text-anchor="middle">Balance L/K ≈ K  ⇒  K ≈ √L</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> at one backward step, the four diamond checkpoints remain live while four discarded activations are rebuilt inside the dashed segment. For this illustrative 16-layer chain, memory is 16/4 + 4 = 8 activation-sized units; shortening segments saves fewer temporary activations but keeps more boundaries, so the terms balance near K = √L.</figcaption>
</figure>

## When to use

- **Always** when training would OOM otherwise.
- **Selectively** for the most memory-intensive blocks (FFN > attention typically). Selective checkpointing recovers most memory at lower compute cost.
- **Less useful** when peak memory is dominated by optimizer state (use [FSDP / ZeRO](/concepts/fsdp-and-zero/) instead).
- **Less useful** at inference (no backward pass).

## Combined with other techniques

- **FSDP**: orthogonal. FSDP shards parameters / gradients / optimizer state; checkpointing reduces activation memory. Most large training runs use both.
- **Mixed precision**: orthogonal; checkpointing saves activations in whatever precision they were computed.
- **CPU offload**: offload activations to CPU memory instead of recomputing. Saves GPU memory at higher communication cost.

## Common pitfalls

- **Recomputing through randomness.** Forward passes with dropout or other stochastic ops must use the same RNG state at recomputation; PyTorch's checkpoint utility handles this with `preserve_rng_state=True` (default).
- **Checkpointing too aggressively.** Larger rematerialized regions save more boundary state and repeat more work. Profile selective, per-block, and larger-segment choices under the real memory limit.
- **Forgetting that the recomputation runs inside the backward graph.** Custom forward hooks may fire twice; gradients stay correct.
- **Trying to checkpoint inference.** Checkpointing only helps when there is a backward pass to run.

## Related

- [FSDP and ZeRO](/concepts/fsdp-and-zero/). For sharding optimizer state and parameters.
- [Mixed precision training](/concepts/mixed-precision-training/). Independent memory reduction.
- [Gradient accumulation](/concepts/gradient-accumulation/). Simulate larger batches without growing per-step activation memory.
