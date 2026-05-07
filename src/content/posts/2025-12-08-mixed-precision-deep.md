---
title: "Mixed precision: what's actually happening?"
description: "Beyond 'use BF16'. The senior answer explains what stays in FP32, why loss scaling exists for FP16, and the memory split."
date: "2025-12-08"
draft: false
tags: ["interviews"]
category: "interviews"
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

*Related: [Mixed precision training](/reference/mixed-precision-training/), [Walk me through how you'd train a 100B parameter model](/interviews/train-100b-model/), [Quantization](/reference/quantization/).*
