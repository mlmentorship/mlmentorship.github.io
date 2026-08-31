---
title: "Gradient accumulation"
description: "Run several forward-backward passes before each optimizer step to simulate a larger effective batch size without the memory cost."
date: "2026-02-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Gradient accumulation runs $K$ forward-backward passes on $K$ micro-batches, summing (or averaging) the gradients across those passes, and then performs a single optimizer step. The effective batch size is $K \times$ the per-pass batch size, with no extra activation memory.

Many training recipes prescribe a specific effective batch size (e.g., 1024 sequences) for stable convergence. If your GPU can only hold 32 sequences at a time, you have two choices: spread the batch across 32 GPUs, or accumulate gradients over 32 steps on one GPU.

Gradient accumulation is the cheap option. It is the standard way to (a) match published training recipes on smaller hardware and (b) increase effective batch size in fine-tuning loops where multi-GPU is not available.

## The mechanism

Replace this loop:

```
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

With:

```
for step, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps   # average over K micro-batches
    loss.backward()                             # gradients accumulate in .grad
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Key points:

- `loss.backward()` adds to existing `.grad` (does not overwrite). Calling it $K$ times accumulates.
- Divide the loss by $K$ so the accumulated gradient equals the average over the effective batch (matches a single big batch's gradient).
- Optimizer step happens once per $K$ micro-batches.

<!-- visual:gradient-accumulation-state-boundary -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="gradient-accumulation-title">
	<p class="visual-kicker">One accumulation window · K = 4</p>
	<p class="visual-title" id="gradient-accumulation-title">The gradient buffer changes four times; the parameters change once.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 354" role="img" aria-labelledby="gradient-accumulation-svg-title gradient-accumulation-svg-desc">
			<title id="gradient-accumulation-svg-title">Gradient and parameter state across four accumulated micro-batches</title>
			<desc id="gradient-accumulation-svg-desc">Four numbered micro-batches run from left to right using the same parameter value theta t. Each backward pass adds one quarter of its gradient to the gradient buffer, which progresses from g1 over 4 to the sum of g1 through g4 over 4. A vertical boundary follows the fourth pass. Only then does optimizer step change the parameters from theta t to theta t plus 1, after which zero grad clears the buffer.</desc>
			<defs>
				<marker id="gradient-accumulation-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path class="viz-arrow-forward" d="M0 0 L10 5 L0 10 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="18" y="24">FORWARD + BACKWARD ON FOUR EQUAL-SIZED MICRO-BATCHES</text>
			<g aria-label="Four sequential micro-batches">
				<rect class="viz-node viz-node--input" x="18" y="43" width="126" height="48" rx="5"></rect><text class="viz-node-label" x="81" y="64">1 · micro-batch B1</text><text class="viz-node-value" x="81" y="81">loss / 4, then backward</text>
				<rect class="viz-node viz-node--input" x="167" y="43" width="126" height="48" rx="5"></rect><text class="viz-node-label" x="230" y="64">2 · micro-batch B2</text><text class="viz-node-value" x="230" y="81">loss / 4, then backward</text>
				<rect class="viz-node viz-node--input" x="316" y="43" width="126" height="48" rx="5"></rect><text class="viz-node-label" x="379" y="64">3 · micro-batch B3</text><text class="viz-node-value" x="379" y="81">loss / 4, then backward</text>
				<rect class="viz-node viz-node--input" x="465" y="43" width="126" height="48" rx="5"></rect><text class="viz-node-label" x="528" y="64">4 · micro-batch B4</text><text class="viz-node-value" x="528" y="81">loss / 4, then backward</text>
			</g>
			<g aria-label="Backward calls add to the gradient buffer">
				<path class="viz-backward" d="M81 92 V122"></path><path class="viz-backward" d="M230 92 V122"></path><path class="viz-backward" d="M379 92 V122"></path><path class="viz-backward" d="M528 92 V122"></path>
				<rect class="viz-node viz-node--focus" x="18" y="124" width="126" height="54" rx="5"></rect><text class="viz-node-value" x="81" y="145">.grad after backward</text><text class="viz-node-label" x="81" y="166">g1 / 4</text>
				<rect class="viz-node viz-node--focus" x="167" y="124" width="126" height="54" rx="5"></rect><text class="viz-node-value" x="230" y="145">.grad after backward</text><text class="viz-node-label" x="230" y="166">(g1 + g2) / 4</text>
				<rect class="viz-node viz-node--focus" x="316" y="124" width="126" height="54" rx="5"></rect><text class="viz-node-value" x="379" y="145">.grad after backward</text><text class="viz-node-label" x="379" y="166">(g1 + g2 + g3) / 4</text>
				<rect class="viz-node viz-node--focus" x="465" y="124" width="126" height="54" rx="5"></rect><text class="viz-node-value" x="528" y="145">complete average gradient</text><text class="viz-node-label" x="528" y="166">(g1 + g2 + g3 + g4) / 4</text>
			</g>
			<path class="viz-operating-guide" d="M607 34 V326"></path>
			<text class="viz-axis-label" x="617" y="49">BOUNDARY</text>
			<rect class="viz-node viz-node--output" x="617" y="68" width="125" height="54" rx="5"></rect><text class="viz-node-label" x="679.5" y="90">optimizer.step()</text><text class="viz-node-value" x="679.5" y="108">consume average once</text>
			<path class="viz-forward" d="M591 151 H607 V95 H615"></path>
			<text class="viz-axis-label" x="18" y="215">PARAMETER STATE SEEN BY EACH FORWARD PASS</text>
			<g aria-label="All four passes use unchanged parameters theta t">
				<rect class="viz-node" x="18" y="230" width="126" height="42" rx="5"></rect><text class="viz-node-label" x="81" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t</tspan></text>
				<rect class="viz-node" x="167" y="230" width="126" height="42" rx="5"></rect><text class="viz-node-label" x="230" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t</tspan></text>
				<rect class="viz-node" x="316" y="230" width="126" height="42" rx="5"></rect><text class="viz-node-label" x="379" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t</tspan></text>
				<rect class="viz-node" x="465" y="230" width="126" height="42" rx="5"></rect><text class="viz-node-label" x="528" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t</tspan></text>
			</g>
			<path class="viz-baseline" d="M144 251 H167 M293 251 H316 M442 251 H465"></path>
			<path class="viz-forward" d="M591 251 H615"></path>
			<rect class="viz-node viz-node--output" x="617" y="230" width="125" height="42" rx="5"></rect><text class="viz-node-label" x="679.5" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t+1</tspan></text>
			<path class="viz-baseline" d="M18 298 H591"></path>
			<text class="viz-callout" x="304.5" y="293" text-anchor="middle">no optimizer step · parameters stay fixed</text>
			<text class="viz-label" x="18" y="323">zero_grad() starts the window</text>
			<text class="viz-label" x="742" y="323" text-anchor="end">zero_grad() clears .grad for the next window</text>
			<text class="viz-callout" x="380" y="347" text-anchor="middle">effective batch = 4 micro-batches · parameter updates = 1</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> scan the two rows together. Each numbered backward call adds one normalized contribution to <code>.grad</code>, but all four forward passes still use <code>&theta;<sub>t</sub></code>. Only the completed average crosses the dashed boundary into <code>optimizer.step()</code>, producing <code>&theta;<sub>t+1</sub></code> once; <code>zero_grad()</code> then starts a new window. This original schematic assumes four equal-sized micro-batches.</figcaption>
</figure>

## Cost model

- **Activation memory**: same as one micro-batch (each is forward-backward'd independently).
- **Optimizer memory**: unchanged (no extra optimizer state).
- **Wall clock**: an optimizer step runs $K$ forward-backward micro-batches. Per-sample throughput may stay similar, improve through less frequent communication, or fall because smaller matrix shapes use the accelerator poorly. Measure it.
- **Convergence**: nearly equivalent to a true large-batch step, modulo BatchNorm (see pitfalls).

## Combined with other techniques

- **DDP / FSDP**: gradient accumulation reduces the frequency of inter-GPU communication. With $N$ GPUs and $K$ accumulation steps, only one all-reduce per $K$ micro-batches → faster throughput. PyTorch's `model.no_sync()` skips the all-reduce on intermediate steps.
- **Mixed precision**: works identically; the loss scaler handles accumulated gradients.
- **Activation checkpointing**: orthogonal; combine for maximum effective batch size on minimum memory.

## Common pitfalls

- **Forgetting to divide by $K$.** Without normalization, the gradient magnitude is $K$× larger than expected → effectively a $K$× larger LR.
- **Calling `optimizer.step()` every micro-batch.** Defeats the purpose.
- **Mixing with BatchNorm.** BN computes statistics within a single forward pass. With accumulation, BN sees only the micro-batch. Statistics are noisier than at the effective batch size. Use LayerNorm or GroupNorm instead, or sync BN across micro-batches.
- **Using accumulation as a replacement for distributed training when bandwidth is available.** Multi-GPU with proper sharding is faster and more memory-efficient than serial accumulation.

## Related

- [Activation checkpointing](/concepts/activation-checkpointing/). Independent memory reduction.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Distributed memory reduction.
- [Mixed precision training](/concepts/mixed-precision-training/). Orthogonal memory reduction.
