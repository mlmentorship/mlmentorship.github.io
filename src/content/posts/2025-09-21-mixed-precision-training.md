---
title: "Mixed precision training: FP16, BF16, and FP8"
description: "How modern transformers train at 2-4× the throughput of FP32 without quality loss. The bit layouts matter; the loss-scaling recipe matters more."
date: "2025-09-21"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Mixed precision training computes most operations in lower-precision formats (FP16, BF16, FP8) for speed and memory savings, while keeping a master copy of weights and certain operations in FP32 for numerical stability.

Tensor cores on modern GPUs (A100, H100, B200) execute lower-precision matmuls several times faster than FP32 with proportionally less memory. Mixed precision is the default for large models in 2026.

## The bit layouts

Three formats matter:

| Format | Sign | Exponent | Mantissa | Range | Notes |
|---|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | ±3.4&times;10&#8311;&#8309; | Reference precision |
| FP16 | 1 | 5 | 10 | ±6.5&times;10&#8308; | Limited range; needs loss scaling |
| BF16 | 1 | 8 | 7 | ±3.4&times;10&#8311;&#8309; | Range = FP32, less precision |
| FP8 (E4M3) | 1 | 4 | 3 | ±~448 | Forward-pass only typically |
| FP8 (E5M2) | 1 | 5 | 2 | ±~57344 | Wider range; for backward |

### FP16

5-bit exponent → small dynamic range (10&#8315;&#8309; to ~6.5&times;10&#8308;). Underflows easily, small gradients become 0, large activations become inf. Requires loss scaling.

### BF16

8-bit exponent (same as FP32) → same dynamic range as FP32. 7-bit mantissa, lower precision than FP16 but rarely matters for deep learning. Available on A100 and later, all modern TPUs. **The recommended format for transformer training in 2026.**

### FP8

Available on H100, B100/B200. Two variants: E4M3 (more mantissa, less range, for forward) and E5M2 (more range, less mantissa, for backward gradients). Used for LLM training at extreme scale; requires careful handling.

## The FP16 recipe

The original mixed-precision recipe [(Micikevicius et al. 2018)](https://arxiv.org/abs/1710.03740):

1. Master weights in FP32.
2. Cast weights to FP16 for forward pass.
3. Compute forward + backward in FP16. Activations and gradients are FP16.
4. **Loss scaling**: multiply the loss by S (e.g., 2&#185;&#8309;) before backward. This shifts gradients up into FP16's representable range, preventing underflow.
5. Before the optimizer step: cast gradients to FP32 and divide by S (unscale).
6. Apply update to FP32 master weights.

<!-- visual:fp16-loss-scaling-ledger -->
<figure class="learning-figure" aria-labelledby="fp16-loss-scaling-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="fp16-loss-scaling-title">Follow one tiny gradient through scaling, FP16 backward, and FP32 unscaling.</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison of the same gradient without loss scaling and with loss scaling">
		<section class="visual-panel" aria-labelledby="unscaled-gradient-title">
			<h4 id="unscaled-gradient-title">Without scaling: the update disappears</h4>
			<table class="cm-grid" aria-label="Unscaled FP16 gradient path">
				<thead><tr><th scope="col">Stage</th><th scope="col">Value</th><th scope="col">Result</th></tr></thead>
				<tbody>
					<tr><th scope="row">True gradient</th><td>g = 2<sup>−30</sup></td><td>Smaller than FP16's minimum nonzero 2<sup>−24</sup></td></tr>
					<tr><th scope="row">FP16 backward</th><td class="cm-selected">store(g) = 0</td><td>Underflow loses the gradient</td></tr>
					<tr><th scope="row">FP32 optimizer</th><td>update = 0</td><td>Master weight does not move</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Casting zero to FP32 cannot recover what FP16 already lost.</p>
		</section>
		<section class="visual-panel" aria-labelledby="scaled-gradient-title">
			<h4 id="scaled-gradient-title">With scaling: enlarge, store, restore</h4>
			<table class="cm-grid" aria-label="Loss-scaled FP16 gradient path">
				<thead><tr><th scope="col">Stage</th><th scope="col">Value</th><th scope="col">Result</th></tr></thead>
				<tbody>
					<tr><th scope="row">Scale loss</th><td>S = 2<sup>20</sup></td><td>Chain rule scales every gradient by S</td></tr>
					<tr><th scope="row">FP16 backward</th><td class="cm-selected">Sg = 2<sup>−10</sup></td><td>Normal, exactly representable FP16 value</td></tr>
					<tr><th scope="row">Cast + unscale</th><td>2<sup>−10</sup> / 2<sup>20</sup> = 2<sup>−30</sup></td><td>Original gradient restored in FP32</td></tr>
					<tr><th scope="row">Clip, then step</th><td>update uses g</td><td>FP32 master weight receives the update</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Scale before backward; unscale before clipping or stepping.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the FP16 backward rows. Without scaling, 2<sup>−30</sup> becomes zero and stays zero after casting. Multiplying the loss by 2<sup>20</sup> makes the backward gradient 2<sup>−10</sup>, which FP16 can store. Convert that value to FP32 and divide by the same scale before clipping or updating: the optimizer receives the original 2<sup>−30</sup>. This original ledger follows <a href="https://openreview.net/forum?id=r1gs9JgRZ">Micikevicius et al.</a> and the <a href="https://docs.pytorch.org/docs/stable/notes/amp_examples.html">PyTorch AMP sequence</a>.</figcaption>
</figure>

**Dynamic loss scaling**: start with large S (e.g., 2&#185;&#8309;). If any gradient is inf/NaN this step, skip the step and halve S. If N consecutive steps go fine, double S. Standard in PyTorch's `torch.cuda.amp.GradScaler`.

## The BF16 recipe

Simpler than FP16:

1. Master weights in FP32.
2. Cast to BF16 for forward.
3. Forward + backward in BF16.
4. Cast gradients to FP32 for the optimizer step.
5. Apply update.

**No loss scaling needed** because BF16 has FP32's dynamic range. This is the main practical advantage of BF16 over FP16.

## The FP8 recipe (advanced)

Used in some frontier LLM training (H100 era). Key elements:

- Per-tensor scaling factors that get updated during training.
- E4M3 for activations and weights in forward; E5M2 for gradients.
- Some operations (LayerNorm, softmax, loss) still run in higher precision.
- Frameworks like Transformer Engine (NVIDIA) handle the bookkeeping.

For most teams in 2026, FP8 is an optimization for very large training runs (10B+ parameters); BF16 is sufficient for most use cases.

## What stays in FP32

Some operations are kept in FP32 for stability even in mixed precision:

- **LayerNorm / RMSNorm**: variance computation needs the precision; small numerical errors compound.
- **Softmax**: in attention specifically. Standard pattern: cast attention scores to FP32, softmax in FP32, cast back to BF16/FP16 for the matmul with V.
- **Loss function**: usually in FP32.
- **Optimizer state**: Adam moments in FP32 (this dominates memory cost).

Frameworks (PyTorch's autocast, JAX's jax.numpy.bfloat16) handle most of this automatically; you just enable mixed precision and they cast operations as appropriate.

## The memory story

For a 7B model in BF16:
- Model weights: 14 GB
- Gradients: 14 GB
- Adam state (m, v in FP32): 56 GB
- Activations: variable, but often largest
- **Total**: ~84 GB plus activations, fits on a single H100 (80GB) only with careful activation management.

For a 70B model: 10&times; everything → no single GPU. Need FSDP / ZeRO-3 to shard.

The Adam optimizer state in FP32 (m and v) is *the dominant memory consumer* for moderately-sized models in mixed precision. ZeRO Stage 1 shards optimizer state across data-parallel ranks; this is often the biggest single memory win.

## What an interviewer expects you to say

If asked about mixed precision:

1. Distinguish FP16 vs BF16 by exponent/mantissa.
2. State that BF16 is preferred when available (no loss scaling needed).
3. Explain the loss-scaling recipe for FP16.
4. Mention the FP32 master copy + per-tensor casting.
5. Mention which operations stay in FP32 (LayerNorm, softmax, loss).
6. Bonus: mention FP8 for H100-era training; mention optimizer state memory dominance.

## Common confusions

- **"FP16 is faster than BF16."** No, they have the same throughput on tensor cores. BF16 is preferred because it doesn't need loss scaling.
- **"Mixed precision halves my model size."** It halves *activation* size and *gradient* size, but the optimizer state is still FP32 (3x the model size for Adam). Total memory savings are smaller than naive 2x.
- **"BF16 is always strictly better than FP16."** Worse precision can occasionally bite (e.g., BF16 loses precision on values near 0 that FP16 retains). For most transformer training, BF16 wins.
- **"Quantization" and "mixed precision" are the same."** Different. Mixed precision is for *training*. Post-training quantization (INT8, INT4) is for *inference* and changes the actual stored data type, not just the compute precision.

---

*Related: [Adam, AdamW, and modern optimizer choices](/concepts/adam-and-adamw/), [Transformer architecture](/concepts/transformer-architecture/).*
