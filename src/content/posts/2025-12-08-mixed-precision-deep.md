---
title: "Mixed precision: what's actually happening?"
description: "Beyond 'use BF16'. The senior answer explains what stays in FP32, why loss scaling exists for FP16, and the memory split."
date: "2025-12-08"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at LLM-team and infra interviews.*

The interviewer wants more than "use AMP." The L6 answer explains the recipe: master weights in FP32, compute in BF16/FP16, certain ops kept in FP32 for stability, loss scaling for FP16.

## What an L4 answer sounds like

> "Mixed precision uses FP16 or BF16 instead of FP32, which is faster and uses less memory. PyTorch has `torch.cuda.amp` for it."

True, no mechanism. You've turned it on, never debugged it.

## What an L5 answer sounds like

> "Mixed precision computes most operations in lower-precision (BF16 on A100/H100, or FP16 on V100) while keeping a master copy of weights in FP32 for the optimizer step. The recipe:
>
> 1. **Master weights in FP32.** Cast a BF16/FP16 view for the forward pass.
> 2. **Forward + backward in BF16/FP16.** Activations and gradients are low-precision.
> 3. **Cast gradients back to FP32** before the optimizer step.
> 4. **Apply update to FP32 master weights.**
>
> For FP16, add **loss scaling**: multiply the loss by S (e.g., 2^15) before backward to push small gradients above FP16 underflow, then unscale before the optimizer step. BF16 doesn't need loss scaling because it has the same exponent range as FP32.
>
> Some operations stay in FP32 for stability:
> - **LayerNorm/RMSNorm**: variance computation needs precision.
> - **Softmax**: especially in attention; cast scores to FP32, softmax, cast back.
> - **Loss function**: usually FP32.
>
> Frameworks (PyTorch autocast) handle most of the casting automatically."

This is L5. You've described the full recipe with the stability ops named.

**Learning objective:** trace one FP16 training step in the order that preserves small gradients without letting scaled or non-finite values corrupt the FP32 parameters.

<!-- visual:fp16-safe-update-order -->
<figure class="learning-figure plot-panel" aria-labelledby="fp16-update-title">
	<p class="visual-kicker">One FP16 training step</p>
	<p class="visual-title" id="fp16-update-title">Low-precision compute is temporary; the guarded update is FP32.</p>
	<svg viewBox="0 0 360 530" role="img" aria-labelledby="fp16-update-svg-title fp16-update-svg-desc">
		<title id="fp16-update-svg-title">Safe operation order for one FP16 mixed-precision training step</title>
		<desc id="fp16-update-svg-desc">Persistent FP32 parameters feed an autocast forward pass in which eligible operations use FP16 and sensitive operations use FP32. The FP32 loss is multiplied by scale S before backward so small FP16 gradients survive. Gradients are converted to FP32 and divided by S before gradient clipping. A finite-value check then branches: non-finite gradients skip the optimizer update and reduce S, while finite unscaled gradients update the FP32 parameters and may eventually allow S to grow. The next step begins from the updated FP32 parameters.</desc>
		<defs>
			<marker id="fp16-update-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-edge)"></path></marker>
		</defs>
		<text class="viz-axis-label" x="18" y="22">PERSISTENT STATE</text>
		<rect class="viz-node viz-node--state" x="44" y="36" width="272" height="48" rx="4"></rect>
		<text class="viz-callout" x="180" y="57" text-anchor="middle">FP32 parameters + optimizer state</text>
		<text class="viz-label" x="180" y="74" text-anchor="middle">the next valid update lands here</text>
		<path d="M180 84V111" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fp16-update-arrow)"></path>
		<text class="viz-axis-label" x="18" y="110">AUTOCAST COMPUTE</text>
		<rect class="viz-node viz-node--input" x="27" y="124" width="147" height="62" rx="4"></rect>
		<text class="viz-callout" x="101" y="147" text-anchor="middle">eligible ops: FP16</text>
		<text class="viz-label" x="101" y="166" text-anchor="middle">matmul · convolution</text>
		<rect class="viz-node viz-node--focus" x="186" y="124" width="147" height="62" rx="4"></rect>
		<text class="viz-callout" x="260" y="147" text-anchor="middle">sensitive ops: FP32</text>
		<text class="viz-label" x="260" y="166" text-anchor="middle">reductions · loss</text>
		<path d="M101 187V207H180M260 187V207H180V226" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fp16-update-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="44" y="228" width="272" height="48" rx="4"></rect>
		<text class="viz-callout" x="180" y="250" text-anchor="middle">scale loss: L ← S · L</text>
		<text class="viz-label" x="180" y="267" text-anchor="middle">then backward; small FP16 gradients shift up</text>
		<path d="M180 276V302" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fp16-update-arrow)"></path>
		<text class="viz-axis-label" x="18" y="302">GUARD THE FP32 UPDATE</text>
		<rect class="viz-node viz-node--output" x="44" y="316" width="272" height="51" rx="4"></rect>
		<text class="viz-callout" x="180" y="338" text-anchor="middle">cast to FP32 and unscale: g ← g / S</text>
		<text class="viz-label" x="180" y="357" text-anchor="middle">clip only now, using the true gradient magnitude</text>
		<path d="M180 368V393" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fp16-update-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="105" y="395" width="150" height="40" rx="4"></rect>
		<text class="viz-callout" x="180" y="420" text-anchor="middle">all gradients finite?</text>
		<path d="M105 415H71V455M255 415H289V455" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fp16-update-arrow)"></path>
		<text class="viz-callout" x="73" y="407" text-anchor="middle">no</text>
		<text class="viz-callout" x="287" y="407" text-anchor="middle">yes</text>
		<rect class="viz-node viz-node--warning" x="18" y="457" width="143" height="55" rx="4"></rect>
		<text class="viz-callout" x="90" y="479" text-anchor="middle">skip update</text>
		<text class="viz-label" x="90" y="498" text-anchor="middle">reduce S; retry next step</text>
		<rect class="viz-node viz-node--output" x="199" y="457" width="143" height="55" rx="4"></rect>
		<text class="viz-callout" x="271" y="479" text-anchor="middle">optimizer step</text>
		<text class="viz-label" x="271" y="498" text-anchor="middle">update FP32 parameters</text>
		<path d="M271 512V524H349V60H318" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:5 4;marker-end:url(#fp16-update-arrow)"></path>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the center path in order. Scaling protects small FP16 gradients during backward, but it must be undone before clipping or interpreting their magnitude. Only finite, unscaled FP32 gradients may reach the optimizer; an overflow skips the update rather than poisoning the persistent parameters. BF16 normally removes the loss-scaling boxes, not the op-specific autocast or FP32 optimizer-state boundaries. Original schematic checked against <a href="https://arxiv.org/abs/1710.03740">Micikevicius et al. (2018)</a> and the <a href="https://docs.pytorch.org/docs/stable/amp.html">PyTorch AMP documentation</a>.</figcaption>
</figure>

## What an L6 answer sounds like

> "...practical things that bite people:
>
> **The memory savings are smaller than people think.** Activations and gradients shrink 2x in BF16, but optimizer state (Adam m, v in FP32) is 4x the model size and dominates. ZeRO/FSDP sharding is what actually unblocks large models, not mixed precision alone.
>
> **BF16 is preferred over FP16.** Same throughput on tensor cores, no loss scaling needed, no NaN spirals from gradient overflow. FP16 is what you use when you're stuck on V100 or consumer hardware.
>
> **FP8 (H100+) is the next step.** Two variants: E4M3 (more mantissa, less range, for forward) and E5M2 (more range, for backward gradients). Per-tensor scaling factors that update during training. Frameworks like Transformer Engine handle the bookkeeping.
>
> **Mixed precision can introduce silent quality regressions.** Differences appear in long-context attention (numerical stability of softmax over many positions), in very deep networks, and in RL where reward signals are subtle. Always evaluate on the production task, not just on training loss.
>
> **Gradient scaler + grad clipping interact.** If you clip gradients, do it on the unscaled gradients (after `scaler.unscale_()` in PyTorch). Clipping the scaled gradient gives wrong results."

## Tells that get you a strong-hire vote

- You name the **FP32 master weights + low-precision compute** pattern explicitly.
- You explain **why FP16 needs loss scaling** but BF16 doesn't.
- You list ops that **stay in FP32** for stability.
- You acknowledge **optimizer state** as the memory dominator, not weights.
- You bring up **FP8** for H100-era training.

## Tells that get you down-leveled

- "Just use AMP" with no further detail.
- Suggesting FP16 over BF16 on modern hardware.
- No mention of loss scaling for FP16.
- Confusion between mixed precision (training) and quantization (inference).

## Common follow-up

"You said optimizer state dominates memory. Can you mixed-precision the optimizer too?"

The L6 answer:

> "Yes, but carefully. Adam's second moment v is the largest concern; storing it in BF16 loses precision and causes training instability. Approaches: 8-bit Adam (Dettmers et al.) quantizes optimizer state to INT8 with per-block scales, recovering most of the memory with minor quality loss. AdaFactor uses a factored approximation of v with much less memory. For pretraining at scale, ZeRO sharding of FP32 optimizer state across data-parallel ranks is more common than quantizing it."

---

*Related: [Mixed precision training](/concepts/mixed-precision-training/), [Walk me through how you'd train a 100B parameter model](/questions/train-100b-model/), [Quantization](/concepts/quantization/).*
