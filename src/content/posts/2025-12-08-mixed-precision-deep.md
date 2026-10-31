---
title: "Mixed precision: what's actually happening?"
description: "Beyond 'use BF16'. The senior answer explains what stays in FP32, why loss scaling exists for FP16, and the memory split."
date: "2025-12-08"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at LLM-team and infra interviews.*

Explain how mixed-precision training assigns numerical formats while preserving useful gradients and stable optimization.

## Problem

**Context.** A model-training workload seeks the speed or memory benefits of FP16 or BF16 without silently corrupting training.
Activations, reductions, gradients, parameters, and optimizer state can have different numerical requirements.

**Task.** Specify a precision policy, explain range and rounding failure modes, and describe how to validate it against a higher-precision baseline.
Compare the role of loss scaling for FP16 and BF16, including the order of unscaling, clipping, and stepping.

**Constraints.** Distinguish storage dtype, compute dtype, and accumulation dtype; automatic casting does not choose every policy for you.
Account for nonfinite gradients and skipped steps, and do not assume every operation or parameter needs identical precision.

## What an L4 answer sounds like

> "Mixed precision uses FP16 or BF16 instead of FP32, which is faster and uses less memory. PyTorch has `torch.cuda.amp` for it."

True, no mechanism. You've turned it on, never debugged it.

## What an L5 answer sounds like

> "One common mixed-precision recipe keeps FP32 parameters and optimizer state while autocasting eligible operations to BF16 or FP16. The exact policy depends on the device, kernel, framework, and operation:
>
> 1. **Persistent parameters.** FP32 model parameters can already be the authoritative weights; a separate master copy is not always allocated.
> 2. **Operation-specific compute.** Autocast chooses eligible lower-precision operations; it does not make every forward/backward value low-precision.
> 3. **Gradient accumulation.** FP32 leaf parameters normally accumulate FP32 gradients, even though intermediate arithmetic may use lower precision. Casting after underflow cannot recover a lost gradient.
> 4. **Guarded update.** Unscale before clipping, reject non-finite updates, and update the persistent parameters with the intended optimizer-state precision.
>
> FP16 training commonly uses loss scaling before backward to preserve small intermediate gradients, followed by unscaling before the optimizer step. BF16's wider exponent range often removes that need, but finite precision, overflow, and unstable operations still require checks. Preserving a gradient also does not guarantee its weight update is representable; the [mixed-precision mechanism](/concepts/mixed-precision-training/) separates those cases.
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
	<p class="visual-title" id="fp16-update-title">Unscale before clipping; skip nonfinite parameter updates</p><div class="visual-argument" data-visual-argument="fp16-safe-update-order" data-argument-kind="timeline" data-argument-composition="shared-clock"><svg viewBox="0 0 360 459" role="img" aria-label="Unscale before clipping; reject nonfinite updates before stepping. Clipping a scaled gradient uses the wrong effective threshold. A nonfinite result skips the parameter update instead of corrupting the model."><title>Unscale before clipping; reject nonfinite updates before stepping</title><desc>Clipping a scaled gradient uses the wrong effective threshold. A nonfinite result skips the parameter update instead of corrupting the model.</desc><text x="20" y="25" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Scaled backward completes</tspan></text><path d="M30 63L330 63" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="30" cy="63" r="7" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="53" y="94" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="53" dy="0">0</tspan></text><text x="20" y="122" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Unscale and check finite</tspan></text><path d="M30 160L330 160" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="130" cy="160" r="7" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="130" y="191" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-state-stroke)"><tspan x="130" dy="0">1</tspan></text><text x="20" y="219" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Clip the unscaled gradient</tspan></text><path d="M30 257L330 257" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="230" cy="257" r="7" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><text x="230" y="288" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="230" dy="0">2</tspan></text><text x="20" y="316" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Step only when finite</tspan></text><path d="M30 354L330 354" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="330" cy="354" r="7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="307" y="385" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="307" dy="0">3</tspan></text><path d="M30 404L330 404" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M323 399L330 404" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M323 409L330 404" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="430" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="180" dy="0">event order, not elapsed time</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">follow the center path in order. Scaling protects small FP16 gradients during backward, but it must be undone before clipping or interpreting their magnitude. Only finite, unscaled FP32 gradients may reach the optimizer; an overflow skips the update rather than poisoning the persistent parameters. BF16 normally removes the loss-scaling boxes, not the op-specific autocast or FP32 optimizer-state boundaries. Original schematic checked against <a href="https://arxiv.org/abs/1710.03740">Micikevicius et al. (2018)</a> and the <a href="https://docs.pytorch.org/docs/stable/amp.html">PyTorch AMP documentation</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="fp16-safe-update-order"><strong>Check:</strong> Why must clipping use unscaled gradients? Which state changes when a nonfinite update is skipped?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="fp16-safe-update-order">Clipping a scaled gradient uses the wrong effective threshold. A nonfinite result skips the parameter update instead of corrupting the model.</span></figcaption>
</figure>

## What an L6 answer sounds like

> "...practical things that bite people:
>
> **Use a byte ledger.** Two FP32 Adam moment arrays cost eight bytes per parameter; FP32 parameters and their FP32 gradient buffers add four bytes each. Activation savings and temporary casts depend on the implementation. Mixed precision and state sharding address different memory terms.
>
> **BF16 changes range, not mathematical stability.** It often avoids FP16 range failures, but it does not prevent every NaN or guarantee a throughput ordering. Check device support, kernel behavior, reductions, and task quality.
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
