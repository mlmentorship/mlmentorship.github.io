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
	<p class="visual-title" id="gradient-depth-title">Repeated small multipliers erase gradients; large ones amplify them</p><div class="visual-argument" data-visual-argument="gradient-magnitude-through-depth" data-argument-kind="curve"><p class="visual-argument-premise">Repeated local multipliers compound</p><svg viewBox="0 0 360 375" role="img" aria-label="Repeated local multipliers compound. Ten multipliers of 0.8 retain about 0.107 of the signal; ten of 1.2 produce about 6.19. Local slopes determine the product, not depth alone."><title>Repeated local multipliers compound</title><desc>Ten multipliers of 0.8 retain about 0.107 of the signal; ten of 1.2 produce about 6.19. Local slopes determine the product, not depth alone.</desc><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">0</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">10</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">6.5</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-series="Each factor = 0.8" d="M48 193.84615384615384L52.40625 194.74232473921404L56.8125 195.60778801608484L61.21875 196.44359588421838L65.625 197.25076449680205L70.03125 198.0302751881708L74.4375 198.78307566688773L78.84375 199.51008116794284L83.25 200.21217556547202L87.65625 200.89021244734766L92.0625 201.54501615294842L96.46875 202.17738277536932L100.875 202.7880811292907L105.28125 203.37785368568294L109.6875 203.947417474483L114.09375 204.49746495634062L118.5 205.02866486449372L122.90625 205.54166301779657L127.3125 206.03708310588945L131.71875 206.51552744746405L136.125 206.9775777225466L140.53125 207.42379567968925L144.9375 207.85472381892902L149.34375 208.27088605134517L153.75 208.6727883360164L158.15625 209.06091929515264L162.5625 209.435750808149L166.96875 209.79773858528438L171.375 210.14732272176195L175.78125 210.4849282327653L180.1875 210.8109655701808L184.59375 211.12583112161408L189 211.4299076923077L193.40625 211.72356497054565L197.8125 212.0071599771107L202.21875 212.28103749934067L206.625 212.5455305103121L211.03125 212.8009605736598L215.4375 213.04763823452578L219.84375 213.2858633971115L224.25 213.51592568929388L228.65625 213.7381048147469L233.0625 213.95267089299813L237.46875 214.15988478783302L241.875 214.359998424446L246.28125 214.55325509572458L250.6875 214.7398897580386L255.09375 214.9201293168937L259.5 215.0941929027973L263.90625 215.26229213767158L268.3125 215.42463139213785L272.71875 215.581408033985L277.125 215.73281266812407L281.53125 215.87902936832057L285.9375 216.02023590098668L290.34375 216.15660394130478L294.75 216.28829928194585L299.15625 216.41548203463563L303.5625 216.53830682481427L307.96875 216.656922979626L312.375 216.77147470946696L316.78125 216.88210128331252L321.1875 216.98893719803684L325.59375 217.0921123419305L330 217.19175215261538" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></path><path data-proof-series="Each factor = 1.2" d="M48 193.84615384615384L52.40625 193.09037569706607L56.8125 192.31275752463037L61.21875 191.51266820895998L65.625 190.6894583924445L70.03125 189.84245995272704L74.4375 188.97098546045177L78.84375 188.07432762134198L83.25 187.15175870215546L87.65625 186.20252994005148L92.0625 185.22587093489017L96.46875 184.2209890239708L100.875 183.18706863870165L105.28125 182.12327064267936L109.6875 181.02873165064034L114.09375 179.9025633277319L118.5 178.74385166853395L122.90625 177.55165625524631L127.3125 176.32500949443963L131.71875 175.0629158317504L136.125 173.7643509438826L140.53125 172.42826090726038L144.9375 171.05356134265696L149.34375 169.63913653510548L153.75 168.18383852837758L158.15625 166.68648619329485L162.5625 165.14586426911671L166.96875 163.56072237722717L171.375 161.9297740063193L175.78125 160.2516954682547L180.1875 158.52512482374954L184.59375 156.74866077701597L189 154.92086153846157L193.40625 153.04024365452347L197.8125 151.10528080368823L202.21875 149.1144025577193L206.625 147.06599310708754L211.03125 144.95838994956975L215.4375 142.78988254095137L219.84375 140.55871090673773L224.25 138.2630642137475L228.65625 135.9010793004289L233.0625 133.47083916470592L237.46875 130.97037140812702L241.875 128.39764663505412L246.28125 125.75057680559188L250.6875 123.02701354092137L255.09375 120.22474637966187L259.5 117.34150098384643L263.90625 114.37493729305453L268.3125 111.32264762520407L272.71875 108.18215472246118L277.125 104.95090974068195L281.53125 101.62629018075415L285.9375 98.20559776016015L290.34375 94.68605622303365L294.75 91.06480908693254L299.15625 87.33891732449948L303.5625 83.50535697812856L307.96875 79.56101670570192L312.375 75.50269525540446L316.78125 71.32709886756757L321.1875 67.03083860143249L325.59375 62.61042758466442L330 58.06227818338468" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-dasharray:6 3"></path><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">number of repeated factors</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">relative gradient magnitude</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Each factor = 0.8</tspan></text><path d="M20 336L48 336" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="340" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Each factor = 1.2</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> start where all three paths equal 1 at the loss, then move right as backpropagation crosses more layers. A repeated 10% gain or loss looks harmless once, but 50 multiplications separate the gradient by more than four orders of magnitude. Line position, direct labels, and the dashed stable path carry the comparison without color. Original schematic based on the Jacobian analysis in Glorot and Bengio (2010) and Pascanu, Mikolov, and Bengio (2013).</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="gradient-magnitude-through-depth"><strong>Check:</strong> Why does a repeated 0.9 factor become about 0.005 after fifty multiplications?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="gradient-magnitude-through-depth">Ten multipliers of 0.8 retain about 0.107 of the signal; ten of 1.2 produce about 6.19. Local slopes determine the product, not depth alone.</span></figcaption>
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
