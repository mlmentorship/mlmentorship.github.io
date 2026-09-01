---
title: "Pipeline parallelism"
description: "Split the model across GPUs by layer; pipeline mini-batches through the stages. The way to scale across slow interconnects when TP isn't viable."
date: "2026-04-06"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Pipeline parallelism** (PP) splits a model along its depth: GPU 0 holds layers 1–8, GPU 1 holds layers 9–16, etc. A mini-batch is divided into smaller **micro-batches** that flow through the stages so that GPU 0 starts processing micro-batch 2 while GPU 1 processes micro-batch 1, achieving parallel utilization despite the sequential layer dependency.

For large models that do not fit on one accelerator, tensor parallelism is often placed on the fastest links because it communicates every layer. Pipeline parallelism sends activations only between adjacent stages, so it can often use slower links more efficiently. Tensor parallelism can cross nodes when the network and workload support it. The choice follows measured communication cost, not a fixed node boundary.

## The basic idea

Without micro-batches, naive pipeline:

```
GPU 0: forward layer 1-8 ────────────  →  backward layer 1-8 ──── 
GPU 1: ───────────  forward 9-16 ────  →  backward 9-16 ───────
GPU 2: ─────────────────  forward 17-24 → backward 17-24 ─
```

Most GPUs are idle most of the time. The **pipeline bubble**.

With micro-batches, the bubble shrinks:

```
GPU 0: f1  f2  f3  f4 ─────────────────────  b4  b3  b2  b1
GPU 1: ─── f1  f2  f3  f4 ─────────  b4  b3  b2  b1 ───────
GPU 2: ─────── f1  f2  f3  f4  b4  b3  b2  b1 ─────────────
```

Bubble fraction $\approx (\text{stages} - 1) / (\text{stages} + \text{micro-batches} - 1)$.

<!-- visual:pipeline-fill-drain-bubble -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="pipeline-bubble-title">
	<p class="visual-kicker">Learning objective · P = 3 stages · M = 4 micro-batches</p>
	<p class="visual-title" id="pipeline-bubble-title">Count the idle slots forced by filling and draining the pipeline.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 680 302" role="img" aria-labelledby="pipeline-bubble-svg-title pipeline-bubble-svg-desc">
			<title id="pipeline-bubble-svg-title">Three-stage pipeline schedule for four micro-batches</title>
			<desc id="pipeline-bubble-svg-desc">A grid has three stage rows and six time columns. Stage 1 processes micro-batches 1 through 4 in time slots 1 through 4, then has two idle slots. Stage 2 begins with one idle slot, processes micro-batches 1 through 4 in slots 2 through 5, then has one idle slot. Stage 3 begins with two idle slots and processes micro-batches 1 through 4 in slots 3 through 6. There are 12 active cells and 6 dotted idle cells, so the visible bubble is 6 of 18 stage-slots, or one third.</desc>
			<defs>
				<marker id="pipeline-wave-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" style="fill:var(--viz-focus-stroke)"></path></marker>
			</defs>
			<text class="viz-axis-label" x="28" y="43">STAGE</text>
			<text class="viz-axis-label" x="391" y="18" text-anchor="middle">TIME SLOT &#8594;</text>
			<text class="viz-axis-label" x="176" y="43" text-anchor="middle">1</text>
			<text class="viz-axis-label" x="262" y="43" text-anchor="middle">2</text>
			<text class="viz-axis-label" x="348" y="43" text-anchor="middle">3</text>
			<text class="viz-axis-label" x="434" y="43" text-anchor="middle">4</text>
			<text class="viz-axis-label" x="520" y="43" text-anchor="middle">5</text>
			<text class="viz-axis-label" x="606" y="43" text-anchor="middle">6</text>
			<rect class="viz-plot-bg" x="91" y="53" width="558" height="153" rx="5"></rect>
			<path class="viz-gridline" d="M91 104H649M91 155H649M133 53V206M219 53V206M305 53V206M391 53V206M477 53V206M563 53V206M649 53V206"></path>
			<text class="viz-callout" x="28" y="84">S1</text>
			<text class="viz-callout" x="28" y="135">S2</text>
			<text class="viz-callout" x="28" y="186">S3</text>
			<text class="viz-label" x="55" y="84">early</text>
			<text class="viz-label" x="55" y="135">middle</text>
			<text class="viz-label" x="55" y="186">late</text>
			<rect x="140" y="61" width="72" height="35" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
			<rect x="226" y="61" width="72" height="35" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
			<rect x="312" y="61" width="72" height="35" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
			<rect x="398" y="61" width="72" height="35" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
			<rect x="226" y="112" width="72" height="35" rx="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="312" y="112" width="72" height="35" rx="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="398" y="112" width="72" height="35" rx="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="484" y="112" width="72" height="35" rx="4" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="312" y="163" width="72" height="35" rx="4" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect>
			<rect x="398" y="163" width="72" height="35" rx="4" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect>
			<rect x="484" y="163" width="72" height="35" rx="4" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect>
			<rect x="570" y="163" width="72" height="35" rx="4" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect>
			<text class="viz-callout" x="176" y="83" text-anchor="middle">F1</text>
			<text class="viz-callout" x="262" y="83" text-anchor="middle">F2</text>
			<text class="viz-callout" x="348" y="83" text-anchor="middle">F3</text>
			<text class="viz-callout" x="434" y="83" text-anchor="middle">F4</text>
			<text class="viz-callout" x="262" y="134" text-anchor="middle">F1</text>
			<text class="viz-callout" x="348" y="134" text-anchor="middle">F2</text>
			<text class="viz-callout" x="434" y="134" text-anchor="middle">F3</text>
			<text class="viz-callout" x="520" y="134" text-anchor="middle">F4</text>
			<text class="viz-callout" x="348" y="185" text-anchor="middle">F1</text>
			<text class="viz-callout" x="434" y="185" text-anchor="middle">F2</text>
			<text class="viz-callout" x="520" y="185" text-anchor="middle">F3</text>
			<text class="viz-callout" x="606" y="185" text-anchor="middle">F4</text>
			<text class="viz-axis-label" x="520" y="83" text-anchor="middle">&#183; idle &#183;</text>
			<text class="viz-axis-label" x="606" y="83" text-anchor="middle">&#183; idle &#183;</text>
			<text class="viz-axis-label" x="176" y="134" text-anchor="middle">&#183; idle &#183;</text>
			<text class="viz-axis-label" x="606" y="134" text-anchor="middle">&#183; idle &#183;</text>
			<text class="viz-axis-label" x="176" y="185" text-anchor="middle">&#183; idle &#183;</text>
			<text class="viz-axis-label" x="262" y="185" text-anchor="middle">&#183; idle &#183;</text>
			<path d="M176 96L255 112M262 147L341 163" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;marker-end:url(#pipeline-wave-arrow)"></path>
			<text class="viz-label" x="213" y="110" transform="rotate(12 213 110)">F1 advances</text>
			<path class="viz-operating-guide" d="M305 48V211M477 48V211"></path>
			<text class="viz-axis-label" x="219" y="224" text-anchor="middle">FILL · 3 idle slots</text>
			<text class="viz-axis-label" x="391" y="224" text-anchor="middle">FULL · all stages busy</text>
			<text class="viz-axis-label" x="563" y="224" text-anchor="middle">DRAIN · 3 idle slots</text>
			<rect x="91" y="240" width="558" height="46" rx="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect>
			<text class="viz-callout" x="370" y="259" text-anchor="middle">12 active + 6 idle = 18 stage-slots</text>
			<text class="viz-axis-label" x="370" y="277" text-anchor="middle">bubble = 6 / 18 = 1 / 3 = (P &#8722; 1) / (M + P &#8722; 1)</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow F1 diagonally: each micro-batch must finish an earlier stage before entering the next, so later stages idle while the pipeline fills and earlier stages idle while it drains. Here 12 useful cells and 6 dotted idle cells occupy 18 stage-slots, making the bubble fraction 6/18 = 1/3. Increasing M extends the full middle without adding more fill or drain depth.</figcaption>
</figure>

## GPipe vs. 1F1B vs. interleaved

- **GPipe** [(Huang et al., 2018)](https://arxiv.org/abs/1811.06965): all forwards then all backwards. Bubble fraction high; activation memory high (must store all forwards).
- **1F1B** (one forward, one backward; PipeDream): start backward as soon as the first micro-batch reaches the last stage. For fixed stages and micro-batches, it has the same basic fill/drain bubble as GPipe but lowers peak activation memory by releasing each micro-batch sooner.
- **Interleaved 1F1B** (Megatron): each GPU holds non-contiguous chunks of layers (e.g., layers 1-2 and 9-10) so the bubble shrinks further.
- **Zero Bubble Pipeline** (recent): split backward into weight-grad and input-grad parts to fill almost all bubbles.

Large training systems may use interleaved 1F1B or zero-bubble schedules when simple schedules leave too much idle time.

## Cost model

For a model with $L$ layers split across $P$ stages and $M$ micro-batches:

- **Bubble**: $(P - 1) / (P + M - 1)$ of total time. Minimize by increasing $M$.
- **Communication per micro-batch**: send activations between adjacent stages. Cost scales with micro-batch × sequence × hidden size. Whether this is cheaper than another layout depends on message frequency, dtype, topology, and overlap.
- **Activation memory per stage**: in 1F1B, $\sim P$ micro-batches' worth of activations.

## When PP wins

- **Cross-node scaling** with slow interconnect.
- **Very deep models** where one stage easily fits on a node.
- **Frontier training** combining 3D parallelism (DP + TP + PP).

## When PP loses

- **Small models** that fit on one node. TP within node + DP across nodes is simpler.
- **Few micro-batches** in a step. The bubble dominates.
- **Workloads with very different per-layer compute**. Load imbalance creates idle GPUs.

## 3D parallelism

One common large-model layout combines:

- **Tensor parallelism** on the fastest useful links.
- **Pipeline parallelism** across balanced groups of layers.
- **Data parallelism or FSDP** across the remaining replica dimension.

The degrees must multiply to the available device count, but that arithmetic does not select the layout. Memory, global batch, topology, and measured scaling select it.

## Common pitfalls

- **Few micro-batches create a large bubble.** More micro-batches improve utilization, but they interact with activation memory and the global batch.
- **Imbalanced stages.** Embeddings, output layers, and uneven blocks can make one stage slower than the others.
- **Forgetting activation memory.** The schedule controls how many micro-batches' activations remain live. Combine PP with activation checkpointing when needed.
- **Treating PP as the same as TP.** They split different axes and communicate at different points. Compare their actual activation messages and schedule overhead.
- **Using a fixed topology rule.** A node boundary is not a law. Measure the links and collectives on the target cluster.
- **Skipping schedule comparison at many stages.** Interleaving can reduce bubbles, but it changes communication and model placement. Measure both schedules.

## Related

- [Tensor parallelism](/concepts/tensor-parallelism/). Orthogonal sharding within a layer.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Orthogonal sharding for memory.
- [Activation checkpointing](/concepts/activation-checkpointing/). Reduce per-stage activation memory.
- [Strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/). Decide whether another pipeline stage helps.
- [Accelerator network topology](/concepts/accelerator-network-topology/). Place stage boundaries on the target cluster.
