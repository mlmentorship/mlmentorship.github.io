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
	<p class="visual-title" id="relu-init-title">Under these assumptions, variance 2/fan-in preserves the ReLU second moment</p><div class="visual-argument" data-visual-argument="relu-second-moment-through-depth" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 475" role="img" aria-label="The ReLU second moment depends on the initialization scale. Under the stated independence and symmetry assumptions, variance 2/fan-in offsets the half-moment loss of ReLU. This is a moment calculation, not universal gradient stability."><title>The ReLU second moment depends on the initialization scale</title><desc>Under the stated independence and symmetry assumptions, variance 2/fan-in offsets the half-moment loss of ReLU. This is a moment calculation, not universal gradient stability.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">0</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">10</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">1.1</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="8" d="M273.6 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><text x="273.6" y="245" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="273.6" dy="0">8</tspan></text><path data-proof-series="Variance 1/fan-in: factor 1/2 per layer" d="M48 65.45454545454547L52.40625 81.31793511339629L56.8125 95.55301980392127L61.21875 108.32693764268276L65.625 119.78967077696745L70.03125 130.0758063671872L74.4375 139.30611681242712L78.84375 147.58897777301115L83.25 155.0216406394857L87.65625 161.69137438843595L92.0625 167.67649023198368L96.46875 173.0472610916637L100.875 177.86674669247867L105.28125 182.1915339647934L109.6875 186.072401447334L114.09375 189.55491549223402L118.5 192.67996527233794L122.90625 195.48424287243242L127.3125 198.0006741012912L131.71875 200.2588050828182L136.125 202.28514916536432L140.53125 204.1034982223765L144.9375 205.735201999448L149.34375 207.19941878766159L153.75 208.513340366451L158.15625 209.69239385709886L162.5625 210.75042285688878L166.96875 211.69984998065848L171.375 212.5518227182003L175.78125 213.31634432006103L180.1875 214.00239124850873L184.59375 214.6180185726903L189 215.17045454545453L193.40625 215.66618547229365L197.8125 216.11103186887254L202.21875 216.51021680133383L206.625 216.86842721178024L211.03125 217.1898689489746L215.4375 217.47831615038834L219.84375 217.7371555554066L224.25 217.96942626998393L228.65625 218.17785544963863L233.0625 218.3648903197495L237.46875 218.5327269091145L241.875 218.68333583413997L246.28125 218.81848543639978L250.6875 218.9397625452292L255.09375 219.0485911091323L259.5 219.14624891476055L263.90625 219.2338825897635L268.3125 219.31252106566535L272.71875 219.38308765883806L273.6 219.3963068181818L277.125 219.44641091141764L281.53125 219.50323431944926L285.9375 219.55422506248274L290.34375 219.59998183711443L294.75 219.6410418864516L299.15625 219.67788730803434L303.5625 219.71095071427777L307.96875 219.74062031189558L312.375 219.76724445994375L316.78125 219.7911357600019L321.1875 219.8125747265159L325.59375 219.83181308039659L330 219.84907670454547" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="8" data-proof-y="0.00390625"><circle cx="273.6" cy="219.3963068181818" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Variance 2/fan-in: factor 1 per layer" d="M48 65.45454545454547L330 65.45454545454547" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="8" data-proof-y="1"><circle cx="273.6" cy="65.45454545454547" r="5" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">layer count</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">relative second moment</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Variance 1/fan-in: factor 1/2</tspan><tspan x="60" dy="18">per layer</tspan></text><text x="60" y="351" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">0.003906 at 8</tspan></text><path d="M20 386L48 386" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="390" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Variance 2/fan-in: factor 1 per</tspan><tspan x="60" dy="18">layer</tspan></text><text x="60" y="431" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="60" dy="0">1 at 8</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> start each path at second moment one. Independent symmetric weights give a ReLU multiplier of fan-in times weight variance divided by two. Circles end at 1/16, squares at one, and triangles at 81/16. The vertical axis uses shared logarithmic spacing. Marker shapes also distinguish paths. Sources are <a href="https://proceedings.mlr.press/v9/glorot10a.html">Glorot and Bengio (2010)</a> and <a href="https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html">He et al. (2015)</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="relu-second-moment-through-depth"><strong>Check:</strong> Why does weight variance 2/fan-in preserve the illustrated ReLU second moment? What assumptions make that recurrence valid?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="relu-second-moment-through-depth">Under the stated independence and symmetry assumptions, variance 2/fan-in offsets the half-moment loss of ReLU. This is a moment calculation, not universal gradient stability.</span></figcaption>
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
