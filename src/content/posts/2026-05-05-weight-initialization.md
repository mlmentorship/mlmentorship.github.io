---
title: "Weight initialization: trace signal scale through a layer"
description: "Derive ReLU second-moment scaling, compare fan-in and fan-out, and recompute the rule when the activation or path structure changes."
date: "2026-05-05"
draft: false
tags: ["concepts"]
category: "concepts"
---

Weight initialization sets a layer's starting scale before data or optimization can correct it.
Which weight variance prevents repeated shrinking or growth through a simple ReLU chain?

## Summary

Track the activation second moment $q=\mathbb E[x_j^2]$, not just centered variance.
For a layer with four inputs and zero bias, compare weight variances $0.25$, $0.5$, and $0.75$.
Under the assumptions below, their per-layer second-moment multipliers are $0.5$, $1$, and $1.5$.
Four layers therefore yield $1/16$, $1$, and $81/16$ from $q_0=1$.

<!-- visual:relu-second-moment-through-depth -->
<figure class="learning-figure" aria-labelledby="relu-init-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="relu-init-title">Under these assumptions, variance 2/fan-in preserves the ReLU second moment</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 270" role="img" aria-labelledby="relu-init-svg-title relu-init-svg-desc">
			<title id="relu-init-svg-title">Activation second moment through four ReLU layers at three initialization scales</title>
			<desc id="relu-init-svg-desc">Under independent zero-mean symmetric preactivations, each ReLU layer multiplies activation second moment by fan-in times weight variance divided by 2. All three paths start at q zero equal to 1. Weight variance 1 over fan-in multiplies by one half per layer and ends at 1 over 16. Kaiming variance 2 over fan-in multiplies by 1 and remains at 1. Variance 3 over fan-in multiplies by three halves and ends at 81 over 16, about 5.06. Circle, square, and triangle markers plus dotted, dashed, and solid paths distinguish the cases without color.</desc>
			<path class="viz-gridline" d="M48 28V226M48 226H305M112 28V226M176 28V226M240 28V226M304 28V226"></path>
			<path class="viz-pr-curve" d="M48 112L112 95.621L176 79.242L240 62.863L304 46.484"></path>
			<path class="viz-operating-guide" d="M48 112H304"></path>
			<path class="viz-baseline" d="M48 112L112 140L176 168L240 196L304 224" style="stroke-dasharray:2 4"></path>
			<g style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"><path d="M48 107L53 117H43Z"></path><path d="M112 90.621L117 100.621H107Z"></path><path d="M176 74.242L181 84.242H171Z"></path><path d="M240 57.863L245 67.863H235Z"></path><path d="M304 41.484L309 51.484H299Z"></path></g>
			<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"><rect x="44" y="108" width="8" height="8"></rect><rect x="108" y="108" width="8" height="8"></rect><rect x="172" y="108" width="8" height="8"></rect><rect x="236" y="108" width="8" height="8"></rect><rect x="300" y="108" width="8" height="8"></rect></g>
			<g style="fill:var(--viz-neutral-bg);stroke:var(--viz-edge);stroke-width:1.5"><circle cx="48" cy="112" r="4"></circle><circle cx="112" cy="140" r="4"></circle><circle cx="176" cy="168" r="4"></circle><circle cx="240" cy="196" r="4"></circle><circle cx="304" cy="224" r="4"></circle></g>
			<text class="viz-axis-label" x="48" y="16">activation second moment q_l / q_0 · log scale</text>
			<text class="viz-callout" x="58" y="54">3 / fan-in: gain 1.5 / layer</text>
			<text class="viz-callout" x="116" y="129">2 / fan-in: gain 1 (Kaiming)</text>
			<text class="viz-callout" x="58" y="218">1 / fan-in: gain 0.5 / layer</text>
			<text class="viz-axis-label" x="314" y="50">5.06</text><text class="viz-axis-label" x="314" y="116">1</text><text class="viz-axis-label" x="314" y="228">0.0625</text>
			<text class="viz-label" x="48" y="244" text-anchor="middle">0</text><text class="viz-label" x="112" y="244" text-anchor="middle">1</text><text class="viz-label" x="176" y="244" text-anchor="middle">2</text><text class="viz-label" x="240" y="244" text-anchor="middle">3</text><text class="viz-label" x="304" y="244" text-anchor="middle">4</text>
			<text class="viz-axis-label" x="176" y="264" text-anchor="middle">ReLU layers traversed</text>
		</svg>
	</div>
	<p>Forward scaling counts four inputs; backward scaling counts eight outputs in the unequal-width example below.</p>
	<figcaption><strong>Read it this way:</strong> start each path at second moment one. Independent symmetric weights give a ReLU multiplier of fan-in times weight variance divided by two. Circles end at 1/16, squares at one, and triangles at 81/16. The vertical axis uses shared logarithmic spacing. Marker shapes also distinguish paths. Sources are <a href="https://proceedings.mlr.press/v9/glorot10a.html">Glorot and Bengio (2010)</a> and <a href="https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html">He et al. (2015)</a>. <span class="visual-reader-check" data-reader-check="relu-second-moment-through-depth"><strong>Check:</strong> Why does weight variance 2/fan-in preserve the illustrated ReLU second moment? What assumptions make that recurrence valid?</span></figcaption>
</figure>

## The variance argument

Let $z=Wx$ with $W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$ and zero bias.
At initialization, weights are mutually independent, zero-mean, symmetric, and independent of the inputs.
Each input coordinate has finite second moment $q$.
Write $v=\operatorname{Var}(W_{ij})$.
Cross terms vanish because distinct weights have zero expected product:

$$
\mathbb E[z_i^2]=\sum_j\mathbb E[W_{ij}^2]\mathbb E[x_j^2]
=d_{\rm in}vq.
$$

This expectation averages the initialized weights as well as the input law.
It is not an exact guarantee for every sampled matrix.
For zero-mean inputs, $q$ equals input variance.

Symmetry of $z_i$ makes each half-line contribute half its second moment.
Therefore

$$
q_{\ell+1}=\mathbb E[\operatorname{ReLU}(z_i)^2]
=\frac{d_{\rm in}v}{2}q_\ell.
$$

Setting $v=2/d_{\rm in}$ preserves this expected second moment: the He/Kaiming fan-in rule.
For $d_{\rm in}=4$, variance $v=0.5$ gives multiplier one.
Variances $0.25$ and $0.75$ give multipliers $0.5$ and $1.5$.

ReLU outputs need not be centered.
If $z$ is equally likely to be $-1$ or $1$, its ReLU second moment is $1/2$.
Its mean is $1/2$ and its centered variance is only $1/4$.

### Change the activation

Leaky ReLU passes negative inputs with slope $a$.
Under symmetry, it retains $(1+a^2)/2$ of the preactivation second moment.
The preserving rule becomes

$$
v=\frac{2}{(1+a^2)d_{\rm in}}.
$$

At $a=1$, the activation is linear and this becomes $1/d_{\rm in}$.
At $a=0.1$ and fan-in four, it gives $v=50/101\approx0.49505$.
Recompute the activation factor rather than treating GELU or sigmoid as exact ReLU cases.

### Change the path structure

Adding a centered branch $F$ to stream $x$ requires covariance accounting:

$$
\operatorname{Var}(x+\alpha F)
=\operatorname{Var}(x)+\alpha^2\operatorname{Var}(F)
+2\alpha\operatorname{Cov}(x,F).
$$

Independent unit-variance stream and branch give variance two when $\alpha=1$.
If $F=x$, the same addition gives variance four, not two.
For $L$ mutually uncorrelated unit-variance branches, scaling each by $1/\sqrt L$ makes their total added variance one.
Correlated residual branches require their covariance terms; a serial-chain rule alone cannot justify that scaling.

### Change the direction

The linear backward map uses $W^\top$, so its analogous independent-gradient calculation counts $d_{\rm out}$ terms.
With unequal widths, one variance cannot preserve both linear directions exactly.
Glorot's compromise is $v=2/(d_{\rm in}+d_{\rm out})$.
For fan-in four and fan-out eight, $v=1/6$ gives forward factor $2/3$ and backward factor $4/3$.
Real backpropagated gradients can depend on weights; this scale argument is an initialization approximation.

**Reconstruct:** can a rectangular orthogonal map preserve all forward and backward vector norms?

<details><summary>Answer</summary>

No, not in both unequal-dimensional spaces.
Orthonormal columns give $W^\top W=I$, preserving every forward input norm.
Orthonormal rows give $WW^\top=I$, preserving the reverse direction instead.
The other direction generally preserves only the corresponding subspace.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Practical defaults

| Component | Starting point | Limit to check |
| --- | --- | --- |
| ReLU linear or convolution | He scale in the intended fan direction | Correct connectivity and weight layout |
| Near-linear centered activation | Glorot scale | Sigmoid offsets and tanh saturation change propagation |
| GELU | Architecture-tested scale | Its second-moment factor depends on its input law |
| Embedding | Architecture-specific distribution | Vocabulary, width, scaling, and downstream normalization |
| Normalization affine parameters | Often scale one, shift zero | Residual designs sometimes choose another scale |
| Bias | Often zero | Recurrent gates and architectural offsets can justify nonzero values |

Do not infer a framework default from an initializer's name.
Check its version, arguments, fan convention, and resulting bounds.

## Special cases

Residual output scaling is architecture-specific; it depends on branch count and covariance, not only network depth.
Embedding scales such as $0.02$ are model conventions, not universal variance laws.
[Identity initialization for ReLU recurrent networks](https://arxiv.org/abs/1504.00941) changes recurrent products but does not remove temporal state.
Orthogonal recurrent weights preserve a linear multiplication's norm; nonlinear derivatives can still shrink gradients.
Rectangular matrices need the row/column distinction above.

## Common pitfalls

- Balanced expected moments do not prove convergence or control every finite-width realization.
- Nonzero bias changes preactivation symmetry; random weights, not zero bias alone, break neuron symmetry.
- Normalization, optimizer state, residual correlations, and activation saturation lie outside the simple serial-chain calculation.
- A training run succeeding does not establish the assumed propagation law or an optimal scale.

</div>
