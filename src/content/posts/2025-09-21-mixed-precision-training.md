---
title: "Mixed precision training: FP16, BF16, and FP8"
description: "Track tensor precision through computation, scaling, guarded updates, and optimizer memory without assuming every tensor shares one dtype."
date: "2025-09-21"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Mixed precision uses lower-precision arithmetic where appropriate while retaining enough range and precision for sensitive computations and persistent state.
Track parameter storage, operation inputs, accumulation, saved activations, gradients, and optimizer state separately.
Autocast selects eligible operations; it does not make every forward or backward tensor share one dtype.

## The bit layouts

The [floating-point owner](/concepts/floating-point-formats/) explains bit fields, rounding, and representable spacing.
Here the question is when an intermediate loses information before a higher-precision operation can use it.

### FP16

FP16's minimum positive normal is $2^{-14}$; its minimum positive subnormal is $2^{-24}$ under gradual underflow.
Its largest finite value is $65504$.
Loss scaling can protect tiny backward intermediates, but cannot guarantee that all gradients fit one usable range.

### BF16

BF16 has the same exponent-bit count as FP32, giving a similar magnitude range but fewer significand bits.
It directly represents $2^{-30}$, avoiding this fixture's FP16 underflow without a loss scale.
Range, rounding, and hardware/operator support remain separate questions.

### FP8

FP8 variants trade range against precision and need a declared encoding and scaling policy.
For example, E4M3FN has maximum finite magnitude $448$, while E5M2 reaches $57344$.
Neither is a universal dtype for every tensor in a training step.

## The FP16 recipe

A manual two-copy configuration follows the principle in [Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740):

1. Keep an FP32 master parameter copy and the declared optimizer state.
2. Supply lower-precision parameter operands to eligible operations; retain sensitive calculations at higher precision.
3. Multiply a higher-precision scalar loss by $S$ before backward, since $\nabla(SL)=S\nabla L$.
4. Accumulate scaled contributions using the same scale throughout one update.
5. Convert as needed, unscale once, inspect finiteness, and clip only the valid unscaled gradient.
6. Perform the optimizer update on persistent parameters, refresh operand copies, and clear gradients at the intended boundary.

For $g=2^{-30}$ and $S=2^{20}$, FP16 stores $Sg=2^{-10}$ exactly; unscaling in FP32 recovers $g$.
Without scaling, FP16 stores $g$ as zero, which a later cast cannot undo.
At FP32 weight $\theta=2^{-10}$ and learning rate one, the updated weight is $2^{-10}-2^{-30}$.
This equals approximately $0.0009765615686774254$ and remains representable in FP32.

<!-- visual:fp16-loss-scaling-ledger -->
<figure class="learning-figure" aria-labelledby="fp16-loss-scaling-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="fp16-loss-scaling-title">Follow one tiny gradient through scaling, FP16 backward, and FP32 unscaling.</p>
	<p class="cm-equation">One plain SGD step, learning rate 1. Initial FP32 master weight: 2<sup>−10</sup>; no momentum or decay.</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison of the same gradient without loss scaling and with loss scaling">
		<section class="visual-panel" aria-labelledby="unscaled-gradient-title">
			<h4 id="unscaled-gradient-title">Without scaling: the gradient is erased</h4>
			<table class="cm-grid" aria-label="Unscaled FP16 gradient path">
				<thead><tr><th scope="col">Stage</th><th scope="col">Value</th><th scope="col">Result</th></tr></thead>
				<tbody>
					<tr><th scope="row">True gradient</th><td>g = 2<sup>−30</sup></td><td>Smaller than FP16's minimum nonzero 2<sup>−24</sup></td></tr>
					<tr><th scope="row">FP16 backward</th><td class="cm-selected">store(g) = 0</td><td>Underflow loses the gradient</td></tr>
					<tr><th scope="row">FP32 SGD step</th><td>update = 0</td><td>Master weight stays 2<sup>−10</sup></td></tr>
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
					<tr><th scope="row">FP32 SGD step</th><td>2<sup>−10</sup> − 2<sup>−30</sup></td><td>Representable change for this weight</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Scale before backward; unscale before clipping or stepping.</p>
		</section>
	</div>
	<p class="cm-equation">Recovery is not guaranteed movement: at FP32 weight 1, subtracting this same 2<sup>−30</sup> rounds back to 1.</p>
	<figcaption><strong>Read it this way:</strong> compare the FP16 backward rows. Without scaling, 2<sup>−30</sup> becomes zero. Scaling by 2<sup>20</sup> instead stores 2<sup>−10</sup>. Cast to FP32, then divide by the same scale before clipping or stepping. The original gradient survives this path. The ledger follows <a href="https://openreview.net/forum?id=r1gs9JgRZ">Micikevicius et al.</a> and the <a href="https://docs.pytorch.org/docs/stable/notes/amp_examples.html">PyTorch AMP sequence</a>. <span class="visual-reader-check" data-reader-check="fp16-loss-scaling-ledger"><strong>Check:</strong> Why can casting an already-zero FP16 gradient to FP32 not recover it, while casting the scaled nonzero gradient can?</span></figcaption>
</figure>

### Change the Loss Scale

**New task:** Lower $S$ to $2^{16}$ for the same tiny gradient.

**Picture and change:** FP16 stores $Sg=2^{-14}$, its minimum positive normal; FP32 unscaling still recovers $2^{-30}$.

**Check:** With $S=1$, the gradient vanishes again.
For a different gradient $0.125$, scale $2^{20}$ produces $131072$, exceeding FP16's finite range.

**Use when:** Dynamic scaling balances backward underflow and overflow.
A nonfinite step must skip the optimizer update and adjust the scale; this does not automatically replay the batch.
Scaling cannot repair an unscaled forward operation that already overflowed.

## The BF16 recipe

**New task:** Use supported BF16 operations instead of the FP16 path.

**Picture and change:** Remove loss-scale/unscale boxes for this fixture, not sensitive reductions or persistent-state accounting.

**Check:** BF16 represents $2^{-30}$ directly with $S=1$.
At weight one, however, a $2^{-30}$ subtraction still rounds away even in FP32.

**Use when:** BF16 operator support is available and FP16 range is the failure mechanism.
The choice does not prove identical quality, universal speedup, or immunity to overflow.

## What stays in FP32

Sensitive normalization statistics, softmax reductions, loss calculations, or optimizer moments may use FP32 according to the implementation.
Read the operator and optimizer policy rather than inferring storage from an autocast context.
Persistent FP32 parameters with autocast are another configuration; they do not require a separate master copy of those same parameters.

## The memory story

For seven billion parameters, declare separate BF16 parameter/gradient copies, an FP32 master copy, and two FP32 moment tensors.
In decimal GB, their sizes are $14+14+28+56=112$, before activations, temporary operands, communication buffers, or allocator overhead.
This cannot fit in an eighty-decimal-GB budget merely by reducing activations.
Different storage configurations need different ledgers; sharding changes per-rank ownership, not the number of logical model parameters.

<!-- learning:core-end -->
<div data-learning-extension>

## The FP8 recipe (advanced)

Tensor scaling, quantization encodings, observed maxima, and scale histories belong to the FP8 execution policy.
E4M3/E5M2 roles depend on the selected implementation, not a universal forward/backward rule.
Sensitive reductions and persistent state can remain higher precision.
Frameworks such as Transformer Engine manage specific supported policies; benchmark their actual workload and quality effects.

## What an interviewer expects you to say

Declare tensor lifetimes, reconstruct the exact gradient path, explain the nonfinite branch, and compute the chosen memory ledger.
Distinguish information surviving backward from a representable final parameter increment.

## Common confusions

- **Format-only speed claims.** Kernel support, shapes, memory movement, and accumulation rules determine observed throughput.
- **Automatic halving.** Not every state changes dtype; count bytes for each actual copy.
- **Range versus precision.** BF16 has fewer significand bits than FP16 despite its wider exponent range.
- **Workflow confusion.** Quantization can appear during training too; integer post-training quantization is one separate workflow.
- **Mixed accumulation units.** Do not change loss scale or unscale only part of an accumulated update.

## Reconstruct

<details>
<summary>Does casting a zero gradient to FP32 restore it?</summary>

No. Scaling must protect the intermediate before rounding erases it.
Even a recovered gradient can produce a rounded-away update when the persistent parameter's representable spacing is too large.

</details>

---

*Related: [Adam, AdamW, and modern optimizer choices](/concepts/adam-and-adamw/), [Transformer architecture](/concepts/transformer-architecture/).*

</div>
