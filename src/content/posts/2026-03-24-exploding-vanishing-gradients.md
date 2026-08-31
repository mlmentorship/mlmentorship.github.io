---
title: "Exploding and vanishing gradients"
description: "Why deep networks were untrainable before residuals, normalization, and ReLU. The math of gradient magnitudes through depth and the standard fixes."
date: "2026-03-24"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Gradients flowing back through a deep network multiply many Jacobians together. If the average per-layer Jacobian norm is $> 1$, gradient magnitudes grow exponentially with depth (**exploding**); if $< 1$, they shrink to zero (**vanishing**). Either failure mode prevents the early layers from learning.

This was the central obstacle to training deep networks before ~2014. The standard fixes. Careful initialization, normalization layers, residual connections, ReLU activations, gradient clipping. Exist primarily to control gradient magnitudes through depth. Knowing the failure mode and the fixes is core senior-level material.

## The math

For a deep net $L \circ f_n \circ f_{n-1} \circ \dots \circ f_1$, the gradient w.r.t. $f_1$ is:

$$
\nabla_{f_1} L = \nabla_{f_n} L \cdot J_{f_n} \cdot J_{f_{n-1}} \cdot \dots \cdot J_{f_2}.
$$

Each $J_{f_i} = \partial f_i / \partial f_{i-1}$. The gradient norm scales roughly as the product of the per-layer Jacobian operator norms. With $n$ layers and average norm $\rho$:

$$
\|\nabla_{f_1} L\| \sim \rho^{n-1}.
$$

Over 50 Jacobian multiplications, $\rho = 1.1$ gives a factor of ~$117$ (gradients explode), while $\rho = 0.9$ gives a factor of ~$0.005$ (gradients vanish).

<!-- visual:gradient-magnitude-through-depth -->
<figure class="learning-figure" aria-labelledby="gradient-depth-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="gradient-depth-title">See how a small per-layer scale error compounds through depth</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 280" role="img" aria-labelledby="gradient-depth-svg-title gradient-depth-svg-desc">
			<title id="gradient-depth-svg-title">Gradient magnitude after repeated Jacobian multiplication</title>
			<desc id="gradient-depth-svg-desc">A logarithmic plot compares three directly labeled trajectories over 50 backward steps. When the representative Jacobian norm rho is 1.1, the relative gradient grows from 1 to 117. When rho is 1, a dashed line stays at 1. When rho is 0.9, the relative gradient shrinks from 1 to 0.005. All three begin at the loss and separate as the gradient moves toward earlier layers.</desc>
			<rect class="viz-plot-bg" x="42" y="22" width="304" height="205" rx="5"></rect>
			<path class="viz-gridline" d="M42 39H346M42 79H346M42 119H346M42 159H346M42 199H346"></path>
			<path class="viz-axis" d="M42 22V227H346M42 227V232M103 227V232M164 227V232M224 227V232M285 227V232M346 227V232"></path>
			<text class="viz-label" x="34" y="43" text-anchor="end">100</text>
			<text class="viz-label" x="34" y="83" text-anchor="end">10</text>
			<text class="viz-label" x="34" y="123" text-anchor="end">1</text>
			<text class="viz-label" x="34" y="163" text-anchor="end">0.1</text>
			<text class="viz-label" x="34" y="203" text-anchor="end">0.01</text>
			<text class="viz-label" x="42" y="247" text-anchor="middle">0</text>
			<text class="viz-label" x="103" y="247" text-anchor="middle">10</text>
			<text class="viz-label" x="164" y="247" text-anchor="middle">20</text>
			<text class="viz-label" x="224" y="247" text-anchor="middle">30</text>
			<text class="viz-label" x="285" y="247" text-anchor="middle">40</text>
			<text class="viz-label" x="346" y="247" text-anchor="middle">50</text>
			<text class="viz-axis-label" x="194" y="269" text-anchor="middle">Jacobians multiplied on the backward path</text>
			<path class="viz-pr-curve" d="M42 119L103 102L164 86L224 69L285 53L346 36"></path>
			<path class="viz-baseline" d="M42 119H346"></path>
			<path class="viz-roc-curve" d="M42 119L103 137L164 156L224 174L285 192L346 211"></path>
			<circle class="viz-operating-point" cx="346" cy="36" r="4"></circle>
			<rect class="viz-node--focus" x="342" y="115" width="8" height="8"></rect>
			<circle class="viz-operating-point" cx="346" cy="211" r="4"></circle>
			<text class="viz-callout" x="336" y="32" text-anchor="end">ρ = 1.1 → 117×</text>
			<text class="viz-callout" x="336" y="112" text-anchor="end">ρ = 1 → 1×</text>
			<text class="viz-callout" x="336" y="222" text-anchor="end">ρ = 0.9 → 0.005×</text>
			<text class="viz-label" x="49" y="16">relative gradient magnitude · log scale</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> start where all three paths equal 1 at the loss, then move right as backpropagation crosses more layers. A repeated 10% gain or loss looks harmless once, but 50 multiplications separate the gradient by more than four orders of magnitude. Line position, direct labels, and the dashed stable path carry the comparison without color. Original schematic based on the Jacobian analysis in Glorot and Bengio (2010) and Pascanu, Mikolov, and Bengio (2013).</figcaption>
</figure>

## Where it shows up

| Architecture | Failure mode |
|--------------|--------------|
| Deep MLP with sigmoid activations | Vanishing (sigmoid derivative ≤ 0.25, multiplies away) |
| RNNs with tanh / sigmoid (long sequences) | Both: gradient through time multiplies $\partial h_{t+1}/\partial h_t$ many times |
| Deep CNN without normalization | Vanishing in early layers |
| Transformers without LayerNorm + residuals | Both: hard to train past 6–12 layers |

## The fixes (and what they actually do)

### Weight initialization
Scale initial weights so that per-layer activation variance is preserved on the forward pass and gradient variance is preserved on the backward. **Kaiming / He init** (for ReLU) and **Xavier / Glorot init** (for tanh) achieve this. Without it, gradients vanish or explode at step 0. See [weight initialization](/concepts/weight-initialization/).

### Non-saturating activations
**ReLU** has gradient exactly 1 in the active region; doesn't shrink gradients through depth (unlike sigmoid / tanh which max out at 0.25). Modern alternatives: GELU, swish (smooth, non-saturating).

### Normalization
**BatchNorm**, **LayerNorm**, **RMSNorm** stabilize activation distributions across layers, indirectly stabilizing gradient magnitudes. LayerNorm in transformers is essential. Without it, deep transformers don't train.

### Residual connections
**Skip connections** ($f(x) + x$) provide a "highway" for gradients to flow back without being attenuated through every layer's Jacobian. Enabled deep ResNets (152+ layers) and made deep transformers practical. The gradient now contains an "identity" term that bypasses each block.

### Gradient clipping
Cap the gradient norm at a fixed threshold ($c = 1.0$ standard for transformers). Doesn't prevent exploding gradients structurally, but keeps any single optimizer step from causing divergence. See [gradient clipping](/concepts/gradient-clipping/).

### Better optimizers
Adam-family optimizers normalize per-parameter gradients by their running variance, partially counteracting magnitude differences across layers.

## Diagnostics

If your deep network won't train:

- Log per-layer **gradient norm** during training. Vanishing: early layers have norm $< 10^{-6}$. Exploding: late layers have norm $> 10^4$.
- Log per-layer **activation magnitudes**. Should be roughly constant across depth; collapsing or exploding indicates trouble.
- Single-batch overfit. If a deep net can't memorize one batch, suspect optimization pathology.

## Common pitfalls

- **Using sigmoid in deep MLP hidden layers.** Vanishing gradients. Use ReLU or GELU.
- **Stacking transformer blocks without LayerNorm.** Deep transformers won't train without it (or RMSNorm).
- **Using PyTorch default `nn.Linear` init for transformers.** Default Kaiming-uniform is wrong scale for typical transformer FFN; many implementations override to $\mathcal{N}(0, 0.02^2)$.
- **Treating gradient clipping as a substitute for proper architecture.** Clipping bounds individual steps but doesn't fix structural vanishing.
- **Assuming residual = problem solved.** Residual connections help enormously but don't substitute for normalization or sensible init.
