---
title: "Weight initialization (Kaiming, Xavier)"
description: "Set the initial variance of each layer's weights so that activations and gradients neither explode nor vanish through depth. The single most impactful one-line decision in deep nets."
date: "2026-05-05"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Initialize each layer's weights from a distribution whose variance is set so the variance of activations (forward pass) and gradients (backward pass) stays approximately constant from layer to layer. Two standard schemes: **Xavier/Glorot** for tanh/sigmoid layers, **Kaiming/He** for ReLU-family layers.

If weights are too small, activations shrink toward zero through depth and gradients vanish. If too large, activations explode and gradients blow up. With either failure, training stalls or diverges in the first few hundred steps.

A correct init lets a 24-layer transformer train to convergence with vanilla SGD or Adam; an incorrect init makes the same architecture untrainable without ad-hoc fixes (warmup hacks, smaller LR, etc.).

<!-- visual:relu-second-moment-through-depth -->
<figure class="learning-figure" aria-labelledby="relu-init-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="relu-init-title">See why ReLU needs the factor of 2</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 270" role="img" aria-labelledby="relu-init-svg-title relu-init-svg-desc">
			<title id="relu-init-svg-title">Activation second moment through four ReLU layers at three initialization scales</title>
			<desc id="relu-init-svg-desc">Under independent zero-mean symmetric preactivations, each ReLU layer multiplies activation second moment by fan-in times weight variance divided by 2. All three paths start at q zero equal to 1. Weight variance 1 over fan-in multiplies by one half per layer and ends at 1 over 16. Kaiming variance 2 over fan-in multiplies by 1 and remains at 1. Variance 3 over fan-in multiplies by three halves and ends at 81 over 16, about 5.06. Circle, square, and triangle markers plus dotted, dashed, and solid paths distinguish the cases without color.</desc>
			<path class="viz-gridline" d="M48 28V226M48 226H305M112 28V226M176 28V226M240 28V226M304 28V226"></path>
			<path class="viz-pr-curve" d="M48 112L112 92L176 72L240 52L304 32"></path>
			<path class="viz-operating-guide" d="M48 112H304"></path>
			<path class="viz-baseline" d="M48 112L112 140L176 168L240 196L304 224" style="stroke-dasharray:2 4"></path>
			<g style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"><path d="M48 107L53 117H43Z"></path><path d="M112 87L117 97H107Z"></path><path d="M176 67L181 77H171Z"></path><path d="M240 47L245 57H235Z"></path><path d="M304 27L309 37H299Z"></path></g>
			<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"><rect x="44" y="108" width="8" height="8"></rect><rect x="108" y="108" width="8" height="8"></rect><rect x="172" y="108" width="8" height="8"></rect><rect x="236" y="108" width="8" height="8"></rect><rect x="300" y="108" width="8" height="8"></rect></g>
			<g style="fill:var(--viz-neutral-bg);stroke:var(--viz-edge);stroke-width:1.5"><circle cx="48" cy="112" r="4"></circle><circle cx="112" cy="140" r="4"></circle><circle cx="176" cy="168" r="4"></circle><circle cx="240" cy="196" r="4"></circle><circle cx="304" cy="224" r="4"></circle></g>
			<text class="viz-axis-label" x="48" y="16">activation second moment q_l / q_0 · log scale</text>
			<text class="viz-callout" x="58" y="54">3 / fan-in: gain 1.5 / layer</text>
			<text class="viz-callout" x="116" y="104">2 / fan-in: gain 1 (Kaiming)</text>
			<text class="viz-callout" x="58" y="218">1 / fan-in: gain 0.5 / layer</text>
			<text class="viz-axis-label" x="314" y="35">5.06</text><text class="viz-axis-label" x="314" y="116">1</text><text class="viz-axis-label" x="314" y="228">0.0625</text>
			<text class="viz-label" x="48" y="244" text-anchor="middle">0</text><text class="viz-label" x="112" y="244" text-anchor="middle">1</text><text class="viz-label" x="176" y="244" text-anchor="middle">2</text><text class="viz-label" x="240" y="244" text-anchor="middle">3</text><text class="viz-label" x="304" y="244" text-anchor="middle">4</text>
			<text class="viz-axis-label" x="176" y="264" text-anchor="middle">ReLU layers traversed</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow each path from the shared second moment <var>q</var><sub>0</sub> = 1. A linear map with <var>fan-in</var> · Var(<var>W</var>) = <var>c</var> multiplies the second moment by <var>c</var>; under the usual symmetric-preactivation assumptions, ReLU keeps half, so each layer multiplies by <var>c</var>/2. The dotted circle path shrinks to 1/16, the dashed square Kaiming path stays at 1, and the solid triangle path grows to 81/16. Position, labels, marker shapes, and line styles carry the result without color. Original schematic based on <a href="https://proceedings.mlr.press/v9/glorot10a.html">Glorot and Bengio (2010)</a> and <a href="https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html">He et al. (2015)</a>.</figcaption>
</figure>

## The variance argument

For a linear layer $y = W x$ with $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, $x$ zero-mean with variance $\sigma_x^2$, and $W$ drawn iid with mean 0 and variance $\sigma_W^2$:

$$
\text{Var}(y_i) = d_\text{in} \cdot \sigma_W^2 \cdot \sigma_x^2.
$$

To preserve variance ($\sigma_y^2 = \sigma_x^2$), pick $\sigma_W^2 = 1 / d_\text{in}$.

The same argument on the backward pass gives $\sigma_W^2 = 1 / d_\text{out}$. Compromise:

$$
\sigma_W^2 = \frac{2}{d_\text{in} + d_\text{out}} \quad \text{(Xavier/[Glorot, 2010](https://proceedings.mlr.press/v9/glorot10a.html))}
$$

For ReLU activations, it is more precise to track the second moment $q = \mathbb{E}[x^2]$: ReLU outputs are not zero-mean, so their variance is not their second moment. Under the usual independent, symmetric-preactivation assumptions, ReLU zeros half the values and retains half the second moment. Compensate with a factor of 2:

$$
\sigma_W^2 = \frac{2}{d_\text{in}} \quad \text{([Kaiming/He, 2015](https://arxiv.org/abs/1502.01852), "fan-in" mode)}
$$

## Practical defaults

| Layer type | Init |
|------------|------|
| Linear, ReLU/GELU activation | Kaiming-normal, fan-in |
| Linear, tanh/sigmoid | Xavier-uniform |
| Conv, ReLU | Kaiming-normal, fan-in |
| Embeddings | $\mathcal{N}(0, 0.02^2)$ for transformers; $\mathcal{N}(0, 1)$ when followed by LayerNorm |
| LayerNorm $\gamma$ | 1 |
| LayerNorm $\beta$ | 0 |
| Bias | 0 |

Most modern frameworks default to Kaiming-uniform for `nn.Linear` (PyTorch). For transformers, GPT-style models often add a per-residual scaling $1/\sqrt{2 \cdot N_\text{layers}}$ on the output projections to keep residual-stream variance bounded with depth.

## Special cases

- **Residual connections**: with N layers, the residual stream's variance grows linearly with depth unless the contributions from each block are downscaled. GPT-2 / GPT-3 scale output projections by $1/\sqrt{N}$.
- **Identity init for recurrent** [(Le et al., 2015)](https://arxiv.org/abs/1504.00941): initialize the recurrent weight matrix to the identity to make RNNs behave like feed-forward at $t=0$.
- **Orthogonal init**: weight matrices initialized to orthogonal matrices preserve norms exactly. Used in some RL policy networks.

## Common pitfalls

- **Using PyTorch's default `nn.Linear` for a transformer without checking it.** The default is Kaiming-uniform with the wrong fan; many transformer codebases override it with $\mathcal{N}(0, 0.02^2)$.
- **Initializing bias to nonzero.** Almost never helps; can break symmetry breaking arguments.
- **Forgetting to scale residual outputs.** Without it, deep transformers produce huge residual-stream values at init.
- **Trusting "it trains" as proof of correct init.** It might converge slower than a properly initialized run.
