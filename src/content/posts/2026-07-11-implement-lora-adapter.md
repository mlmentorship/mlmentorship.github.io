---
title: "Implement a LoRA adapter for a frozen linear layer"
description: "The code is small; the signal is low-rank parameterization, zero-impact initialization, scaling, trainable-state control, and deployment trade-offs."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
aliases: ["LoRA", "low-rank adaptation", "parameter efficient fine tuning", "PEFT"]
---

> Wrap a frozen linear layer with a trainable rank-$r$ update. At initialization, output must exactly match the base layer.

Wrap a frozen linear layer with a trainable low-rank update while preserving the base layer's initial output.

## Problem

**Context.** A base linear weight has shape `(d_out, d_in)`, and inputs end in feature dimension `d_in`.
Trainable factors have shapes `(r, d_in)` and `(d_out, r)`, with positive rank `r` and scale `alpha / r`.

**Task.** Implement the adapter forward path and initialization, returning the same output shape as the base layer.
Show that initialization matches the base output and that adapter parameters can receive gradients while the base remains frozen.

**Constraints.** Preserve the base weight and any base bias; match dtype, device, and arbitrary leading input dimensions.
The initial effective update must be zero. Explain factor initialization and distinguish forward equivalence from every factor receiving a nonzero first-step gradient.

For base weight $W \in \mathbb{R}^{d_{out} \times d_{in}}$, the adapted weight is:

$$
W' = W + \frac{\alpha}{r} B A,
$$

with $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$. The base $W$ remains frozen.

## The implementation contract

1. Reject non-positive rank.
2. Freeze base parameters.
3. Initialize one adapter matrix randomly and the other to zero, so $BA = 0$ initially.
4. Compute the base output plus the scaled low-rank path.
5. Preserve arbitrary leading input dimensions.
6. Ensure gradients reach adapter parameters and not the base.
7. Expose only adapter state when saving an adapter checkpoint.

A direct forward pass is:

```python
base_output = self.base(inputs)
update = (inputs @ self.a.T) @ self.b.T
return base_output + self.scaling * update
```

The order computes the low-rank path without constructing a full $d_{out} \times d_{in}$ update.

## Why initialize one factor to zero

If both factors are random, attaching the adapter changes model behavior before training. If both are zero, both gradients begin at zero because each factor's gradient contains the other factor. Initializing $A$ randomly and $B$ to zero gives exact base behavior while allowing $B$ to receive a gradient on the first step. After $B$ moves, $A$ receives useful gradients.

<p class="visual-kicker">Learning objective</p>
<!-- visual:lora-zero-impact-first-gradient -->
<figure class="learning-figure editorial-illustration" aria-labelledby="lora-zero-impact-first-gradient-edition-title"><p class="visual-title" id="lora-zero-impact-first-gradient-edition-title">A zero adapter output can still give B a nonzero first gradient</p><div class="visual-argument" data-visual-argument="lora-zero-impact-first-gradient" data-argument-kind="ledger"><p class="visual-argument-premise">Zero adapter output does not imply zero learning signal</p><svg viewBox="0 0 360 322" role="img" aria-label="Zero adapter output does not imply zero learning signal. With A=1, B=0, x=2, upstream gradient 3, and scale 1, the initial adapter output is zero. Its first B gradient is 6 while the A gradient is zero."><title>Zero adapter output does not imply zero learning signal</title><desc>With A=1, B=0, x=2, upstream gradient 3, and scale 1, the initial adapter output is zero. Its first B gradient is 6 while the A gradient is zero.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Initial adapter factors</tspan></text><rect data-proof-token="A: 1" x="20" y="58" width="156" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="98" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">A: 1</tspan></text><rect data-proof-token="B: 0" x="180" y="58" width="156" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="258" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">B: 0</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Initial adapter output</tspan></text><rect data-proof-token="B A x" x="20" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="71.33333333333334" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">B A x</tspan></text><rect data-proof-token="=" x="126.66666666666667" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="178" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">=</tspan></text><rect data-proof-token="0" x="233.33333333333334" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="284.6666666666667" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">0</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">First backward gradients</tspan></text><rect data-proof-token="dA: 0" x="20" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="98" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">dA: 0</tspan></text><rect data-proof-token="dB: 6" x="180" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="258" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">dB: 6</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 493" role="img" aria-labelledby="lora-zero-impact-first-gradient-edition-svg-title lora-zero-impact-first-gradient-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="lora-zero-impact-first-gradient-edition-svg-title">A zero initial adapter output does not mean both factors have zero gradient</title><desc id="lora-zero-impact-first-gradient-edition-svg-desc">At initialization A is random and B is zero, so the scaled adapter output sBAx is zero. For upstream gradient g, dL/dA equals s B-transpose g x-transpose and is zero. dL/dB equals s g (Ax)-transpose and is generally nonzero. Frozen W parameters do not block input gradients.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">At initialization: A random, B = 0; scale s</tspan><tspan x="180" dy="19">= alpha/r</tspan></text><g data-editorial-node><rect x="20" y="58" width="140" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="84.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Frozen base W</tspan><tspan x="90" dy="19">Wx</tspan></text></g><g data-editorial-node><rect x="200" y="58" width="140" height="62" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="84.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Adapter</tspan><tspan x="270" dy="19">sBAx = 0</tspan></text></g><path d="M90 120L145 163" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M138.09341813757402,161.86020748484273L145,163L142.22746022427023,156.57247923441733" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M270 120L215 163" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M217.77253977572977,156.57247923441733L215,163L221.90658186242598,161.86020748484273" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="80" y="167" width="200" height="53" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="198.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Initial y = Wx</tspan></text></g><text x="180" y="267" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">First backward pass; g = dL/dy</tspan></text><g data-editorial-node><rect x="20" y="291" width="320" height="55" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="180" y="323.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">dL/dA = s B^T g x^T = 0</tspan></text></g><g data-editorial-node><rect x="20" y="367" width="320" height="55" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="390" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">dL/dB = s g (Ax)^T</tspan><tspan x="180" dy="19">generally nonzero</tspan></text></g><text x="180" y="457" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">W is not updated; input gradients can still</tspan><tspan x="180" dy="19">flow.</tspan></text></svg><div class="visual-derivation-caption">B = 0 makes the initial adapter output and A gradient zero. B can still receive a gradient. Frozen base weights do not detach inputs.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="lora-zero-impact-first-gradient"><strong>Check:</strong> Why can B receive a nonzero first gradient even though the adapter initially adds zero to the output?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="lora-zero-impact-first-gradient">With A=1, B=0, x=2, upstream gradient 3, and scale 1, the initial adapter output is zero. Its first B gradient is 6 while the A gradient is zero.</span></figcaption></figure>

## What an L4 answer sounds like

The candidate adds a full trainable matrix, forgets to freeze the base, or initializes both low-rank factors to zero. They know LoRA means "fewer parameters" but cannot derive shapes or explain why the initial output should match.

## What an L5 answer adds

An L5 candidate gets shapes, scaling, initialization, and trainable state correct. They test:

- exact initial equivalence to the base layer;
- only adapter parameters require gradients;
- output shape for batched and sequence inputs;
- rank and alpha behavior;
- adapter save and reload;
- merge equivalence: explicit $W + (\alpha/r)BA$ matches the unmerged path.

They can calculate trainable parameters:

$$
r(d_{in} + d_{out})
$$

instead of $d_{in}d_{out}$ for a full update.

## What an L6 answer adds

An L6 candidate discusses where adapters attach and why. Attention projections, MLP projections, embeddings, and output heads have different leverage. Rank is a capacity choice, not merely a memory knob.

They cover serving choices:

- merge one adapter into weights for simple dedicated serving;
- keep adapters separate for multi-tenant swapping;
- batch requests with different adapters only if the serving stack supports efficient segmented adapter computation;
- version base and adapter together because an adapter is not portable across arbitrary base checkpoints;
- preserve quantization semantics in QLoRA, where base weights are quantized but adapter computation uses a higher-precision path.

They also resist an overclaim: low-rank updates can approximate many useful adaptations, but "intrinsic dimension is low" is not a guarantee that every behavior change fits a tiny rank.

## Tells that get you a strong-hire vote

- Matrix shapes are derived before code.
- Initial output exactly matches the base.
- Base parameters are frozen and absent from adapter-only checkpoints.
- The low-rank path avoids materializing a full update.
- Parameter count and scaling are explicit.
- Merge and unmerged paths are tested for equivalence.
- Base-version and multi-tenant serving constraints are discussed.

## Tells that get you down-leveled

- Training the base accidentally.
- Initializing both factors to zero.
- Applying the update with transposed shapes by trial and error.
- Saving the full model when the contract asks for an adapter.
- Claiming LoRA always matches full fine-tuning.
- Calling QLoRA a low-precision adapter rather than a quantized frozen base plus trainable adapters.

## Common follow-up

"Can you merge LoRA for inference?"

Yes. Compute $W' = W + (\alpha/r)BA$ once and use the ordinary linear layer. That removes adapter-path overhead but loses cheap adapter swapping and can complicate quantized weights. Validate merged output against the unmerged module before deployment.

Use the [LoRA starter](/prep/labs/implementation/) before reading the forward pass again.

*Related: [fine-tuning, the deep version](/questions/fine-tuning-deep/), [SVD and PCA](/concepts/svd-and-pca/), and [RLHF and DPO](/concepts/rlhf-and-dpo/).*
