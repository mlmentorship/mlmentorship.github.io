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
	<p class="visual-title" id="gradient-depth-title">Repeated small multipliers erase gradients; large ones amplify them</p><div class="visual-argument" data-visual-argument="gradient-magnitude-through-depth" data-argument-kind="curve" data-argument-composition="gain-cascade"><svg viewBox="0 0 360 482" role="img" aria-label="Repeated local multipliers compound. Ten multipliers of 0.8 retain about 0.107 of the signal; ten of 1.2 produce about 6.19. Local slopes determine the product, not depth alone."><title>Repeated local multipliers compound</title><desc>Ten multipliers of 0.8 retain about 0.107 of the signal; ten of 1.2 produce about 6.19. Local slopes determine the product, not depth alone.</desc><text x="20" y="26" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">The same signal passes every local</tspan><tspan x="20" dy="18">gate</tspan></text><text x="20" y="74" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Every gate multiplies by 0.8</tspan></text><path data-picture-gain="0,0" data-picture-before="1" data-picture-after="0.8" d="M28 137L56 137.8V144.2L28 145Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M56 107L56 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,1" data-picture-before="0.8" data-picture-after="0.6400000000000001" d="M56 137.8L84 138.44V143.56L56 144.2Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M84 107L84 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,2" data-picture-before="0.6400000000000001" data-picture-after="0.5120000000000001" d="M84 138.44L112 138.952V143.048L84 143.56Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M112 107L112 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,3" data-picture-before="0.5120000000000001" data-picture-after="0.4096000000000001" d="M112 138.952L140 139.3616V142.6384L112 143.048Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M140 107L140 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,4" data-picture-before="0.4096000000000001" data-picture-after="0.3276800000000001" d="M140 139.3616L168 139.68928V142.31072L140 142.6384Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M168 107L168 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,5" data-picture-before="0.3276800000000001" data-picture-after="0.2621440000000001" d="M168 139.68928L196 139.951424V142.048576L168 142.31072Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M196 107L196 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,6" data-picture-before="0.2621440000000001" data-picture-after="0.20971520000000007" d="M196 139.951424L224 140.1611392V141.8388608L196 142.048576Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M224 107L224 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,7" data-picture-before="0.20971520000000007" data-picture-after="0.1677721600000001" d="M224 140.1611392L252 140.32891136V141.67108864L224 141.8388608Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M252 107L252 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,8" data-picture-before="0.1677721600000001" data-picture-after="0.13421772800000006" d="M252 140.32891136L280 140.463129088V141.536870912L252 141.67108864Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M280 107L280 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="0,9" data-picture-before="0.13421772800000006" data-picture-after="0.10737418240000006" d="M280 140.463129088L308 140.5705032704V141.4294967296L280 141.536870912Z" style="fill:var(--viz-input-stroke);opacity:.65"></path><path d="M308 107L308 175" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="28" y="205" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">1 in</tspan></text><text x="324" y="205" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-input-stroke)"><tspan x="324" dy="0">0.1074 out</tspan></text><text x="20" y="270" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Every gate multiplies by 1.2</tspan></text><path data-picture-gain="1,0" data-picture-before="1" data-picture-after="1.2" d="M28 333L56 332.2V341.8L28 341Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M56 303L56 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,1" data-picture-before="1.2" data-picture-after="1.44" d="M56 332.2L84 331.24V342.76L56 341.8Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M84 303L84 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,2" data-picture-before="1.44" data-picture-after="1.7279999999999998" d="M84 331.24L112 330.088V343.912L84 342.76Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M112 303L112 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,3" data-picture-before="1.7279999999999998" data-picture-after="2.0736" d="M112 330.088L140 328.7056V345.2944L112 343.912Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M140 303L140 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,4" data-picture-before="2.0736" data-picture-after="2.4883199999999994" d="M140 328.7056L168 327.04672V346.95328L140 345.2944Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M168 303L168 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,5" data-picture-before="2.4883199999999994" data-picture-after="2.9859839999999993" d="M168 327.04672L196 325.056064V348.943936L168 346.95328Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M196 303L196 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,6" data-picture-before="2.9859839999999993" data-picture-after="3.583180799999999" d="M196 325.056064L224 322.6672768V351.3327232L196 348.943936Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M224 303L224 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,7" data-picture-before="3.583180799999999" data-picture-after="4.2998169599999985" d="M224 322.6672768L252 319.80073216V354.19926784L224 351.3327232Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M252 303L252 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,8" data-picture-before="4.2998169599999985" data-picture-after="5.1597803519999985" d="M252 319.80073216L280 316.360878592V357.639121408L252 354.19926784Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M280 303L280 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-gain="1,9" data-picture-before="5.1597803519999985" data-picture-after="6.191736422399997" d="M280 316.360878592L308 312.2330543104V361.7669456896L280 357.639121408Z" style="fill:var(--viz-warning-stroke);opacity:.65"></path><path d="M308 303L308 371" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="28" y="401" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">1 in</tspan></text><text x="324" y="401" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-warning-stroke)"><tspan x="324" dy="0">6.192 out</tspan></text><text x="20" y="451" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Ten gates can shrink or</tspan><tspan x="20" dy="23">amplify.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> start where all three paths equal 1 at the loss, then move right as backpropagation crosses more layers. A repeated 10% gain or loss looks harmless once, but 50 multiplications separate the gradient by more than four orders of magnitude. Line position, direct labels, and the dashed stable path carry the comparison without color. Original schematic based on the Jacobian analysis in Glorot and Bengio (2010) and Pascanu, Mikolov, and Bengio (2013).</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="gradient-magnitude-through-depth"><strong>Check:</strong> Why does a repeated 0.9 factor become about 0.005 after fifty multiplications?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="gradient-magnitude-through-depth">Ten multipliers of 0.8 retain about 0.107 of the signal; ten of 1.2 produce about 6.19. Local slopes determine the product, not depth alone. <br><strong>Is depth alone sufficient to say whether this signal grows?</strong></span></figcaption>
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
