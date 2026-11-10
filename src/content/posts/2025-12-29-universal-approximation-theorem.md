---
title: "Universal approximation theorem"
description: "Construct a ReLU approximation with a checked uniform error, then separate existence, training, generalization and extrapolation."
date: "2025-12-29"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Universal approximation establishes that suitable network parameters exist for a stated target class, domain and error measure.
It does not establish that training finds them.

Approximate $f(x)=x^2$ on $K=[0,1]$ using
$g(x)=0.5\operatorname{ReLU}(x)+\operatorname{ReLU}(x-0.5)$.
Can its largest error be below $\varepsilon=0.07$ everywhere, not just at chosen sample points?

<!-- visual:uat-existence-error-band -->
<figure class="learning-figure plot-panel" aria-labelledby="uat-error-band-title">
	<p class="visual-kicker">The quantifiers are the lesson</p>
	<p class="visual-title" id="uat-error-band-title">An explicit approximation inside the band proves existence here, not trainability or generalization</p><div class="visual-argument" data-visual-argument="uat-existence-error-band" data-argument-kind="curve" data-argument-composition="piecewise-fit"><svg viewBox="0 0 360 800" role="img" aria-label="An explicit approximation stays inside an error ceiling. Four straight segments stay close to the curved target throughout this interval. A construction with such a small error demonstrates that a fit exists. It does not establish that training will find it or that it generalizes outside the interval."><title>An explicit approximation stays inside an error ceiling</title><desc>Four straight segments stay close to the curved target throughout this interval. A construction with such a small error demonstrates that a fit exists. It does not establish that training will find it or that it generalizes outside the interval.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Four line segments approximate the</tspan><tspan x="20" dy="18">curved target</tspan></text><path d="M35 353L38.625 352.96171875L42.25 352.846875L45.875 352.65546875L49.5 352.3875L53.125 352.04296875L56.75 351.621875L60.375 351.12421875L64 350.55L67.625 349.89921875L71.25 349.171875L74.875 348.36796875L78.5 347.4875L82.125 346.53046875L85.75 345.496875L89.375 344.38671875L93 343.2L96.625 341.93671875L100.25 340.596875L103.875 339.18046875L107.5 337.6875L111.125 336.11796875L114.75 334.471875L118.375 332.74921875L122 330.95L125.625 329.07421875L129.25 327.121875L132.875 325.09296875L136.5 322.9875L140.125 320.80546875L143.75 318.546875L147.375 316.21171875L151 313.8L154.625 311.31171875L158.25 308.746875L161.875 306.10546875L165.5 303.3875L169.125 300.59296875L172.75 297.721875L176.375 294.77421875L180 291.75L183.625 288.64921875L187.25 285.471875L190.875 282.21796875L194.5 278.8875L198.125 275.48046875L201.75 271.99687500000005L205.375 268.43671874999995L209 264.8L212.625 261.08671875L216.25 257.296875L219.875 253.43046875L223.5 249.48749999999998L227.125 245.46796875L230.75 241.371875L234.375 237.19921875L238 232.95000000000002L241.625 228.62421875L245.25 224.221875L248.875 219.74296875L252.5 215.1875L256.125 210.55546875000002L259.75 205.84687499999998L263.375 201.06171875L267 196.19999999999996L270.625 191.26171875L274.25 186.24687500000002L277.875 181.15546875L281.5 175.9875L285.125 170.74296875L288.75 165.421875L292.375 160.02421875000002L296 154.54999999999998L299.625 148.99921875L303.25 143.371875L306.875 137.66796875L310.5 131.88750000000002L314.125 126.03046874999998L317.75 120.09687500000001L321.375 114.08671874999999L325 108" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path><path d="M35 353L38.625 352.234375L42.25 351.46875L45.875 350.703125L49.5 349.9375L53.125 349.171875L56.75 348.40625L60.375 347.640625L64 346.875L67.625 346.109375L71.25 345.34375L74.875 344.578125L78.5 343.8125L82.125 343.046875L85.75 342.28125L89.375 341.515625L93 340.75L96.625 339.984375L100.25 339.21875L103.875 338.453125L107.5 337.6875L111.125 335.390625L114.75 333.09375L118.375 330.796875L122 328.5L125.625 326.203125L129.25 323.90625L132.875 321.609375L136.5 319.3125L140.125 317.015625L143.75 314.71875L147.375 312.421875L151 310.125L154.625 307.828125L158.25 305.53125L161.875 303.234375L165.5 300.9375L169.125 298.640625L172.75 296.34375L176.375 294.046875L180 291.75L183.625 287.921875L187.25 284.09375L190.875 280.265625L194.5 276.4375L198.125 272.609375L201.75 268.78125L205.375 264.953125L209 261.125L212.625 257.296875L216.25 253.46875L219.875 249.640625L223.5 245.8125L227.125 241.984375L230.75 238.15625L234.375 234.328125L238 230.5L241.625 226.671875L245.25 222.84375L248.875 219.015625L252.5 215.1875L256.125 209.82812500000003L259.75 204.46874999999997L263.375 199.10937500000003L267 193.74999999999997L270.625 188.390625L274.25 183.03125000000003L277.875 177.67187499999997L281.5 172.31250000000003L285.125 166.95312499999997L288.75 161.59375L292.375 156.23437500000003L296 150.87499999999997L299.625 145.51562500000003L303.25 140.15624999999997L306.875 134.796875L310.5 129.43750000000003L314.125 124.07812499999997L317.75 118.71875000000003L321.375 113.35937499999997L325 108" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2"></path><circle cx="35" cy="353" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="107.5" cy="337.6875" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="180" cy="291.75" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="252.5" cy="215.1875" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="325" cy="108" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="20" y="424" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Magnified error: its peak stays below</tspan><tspan x="20" dy="18">0.02</tspan></text><path d="M35 566L325 566" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M35 686L38.625 668.1875L42.25 652.25L45.875 638.1875L49.5 626L53.125 615.6875L56.75 607.25L60.375 600.6875L64 596L67.625 593.1875L71.25 592.25L74.875 593.1875L78.5 596L82.125 600.6875L85.75 607.25L89.375 615.6875L93 626L96.625 638.1875L100.25 652.25L103.875 668.1875L107.5 686L111.125 668.1875L114.75 652.25L118.375 638.1875L122 626L125.625 615.6875L129.25 607.25L132.875 600.6875L136.5 596L140.125 593.1875000000001L143.75 592.25L147.375 593.1875L151 596.0000000000001L154.625 600.6875L158.25 607.25L161.875 615.6875L165.5 626L169.125 638.1875L172.75 652.25L176.375 668.1875L180 686L183.625 668.1875000000002L187.25 652.25L190.875 638.1874999999999L194.5 626L198.125 615.6875L201.75 607.25L205.375 600.6875000000003L209 595.9999999999999L212.625 593.1875000000001L216.25 592.25L219.875 593.1875000000001L223.5 596.0000000000002L227.125 600.6875L230.75 607.25L234.375 615.6875L238 626L241.625 638.1875000000002L245.25 652.25L248.875 668.1875000000002L252.5 686L256.125 668.1875000000002L259.75 652.25L263.375 638.1875000000002L267 626L270.625 615.6875L274.25 607.2500000000003L277.875 600.6874999999997L281.5 595.9999999999999L285.125 593.1874999999998L288.75 592.25L292.375 593.1875000000005L296 595.9999999999999L299.625 600.6875000000003L303.25 607.2499999999997L306.875 615.6875L310.5 626.0000000000006L314.125 638.1875000000002L317.75 652.25L321.375 668.1874999999995L325 686" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2"></path><text x="20" y="739" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Maximum gap stays below 0.02 in this</tspan><tspan x="20" dy="18">interval.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="uat-error-band-svg-title uat-error-band-svg-desc">
		<title id="uat-error-band-svg-title">An explicit two-ReLU approximation to x squared on zero to one</title>
		<desc id="uat-error-band-svg-desc">The solid quadratic x squared is surrounded by a tolerance band of plus or minus 0.07. The dashed ReLU approximation connects its values at zero, one half and one. Its largest vertical error is 0.0625 at one quarter and three quarters. Coordinates use 250 drawing units per input unit and 180 per output unit. The band and both curves represent these numeric values, not schematic deviations.</desc>
		<rect class="viz-plot-bg" x="8" y="30" width="344" height="246" rx="5"></rect>
		<text class="viz-axis-label" x="14" y="18">ON COMPACT K: UNIFORM ERROR &lt; ε</text>
		<path d="M54 222.4Q179 222.4 304 42.4L304 67.6Q179 247.6 54 247.6Z" style="fill:var(--viz-focus-bg);stroke:none"></path>
		<path class="viz-gridline" d="M54 190H304M54 55H304"></path>
		<path d="M54 235Q179 235 304 55" class="viz-roc-curve"></path>
		<path d="M54 235L179 190L304 55" class="viz-pr-curve" stroke-dasharray="7 5"></path>
		<path d="M54 38V250M304 38V250" class="viz-baseline"></path>
		<path d="M46 235H320" class="viz-axis"></path>
		<text class="viz-callout" x="68" y="100">f(x) = x²: solid</text>
		<text class="viz-callout" x="185" y="116">g(x): dashed</text>
		<text class="viz-label" x="68" y="62">band: f(x) ± 0.07</text>
		<text class="viz-label" x="38" y="59">1</text>
		<text class="viz-label" x="25" y="194">0.25</text>
		<text class="viz-label" x="38" y="239">0</text>
		<path d="M116.5 212.5H124M116.5 223.75H124M121 212.5V223.75" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
		<text class="viz-callout" x="126" y="219">error = 0.0625</text>
		<text class="viz-label" x="54" y="254" text-anchor="middle">0</text>
		<text class="viz-label" x="179" y="254" text-anchor="middle">0.5</text>
		<text class="viz-label" x="304" y="254" text-anchor="middle">1</text>
		<text class="viz-axis-label" x="180" y="272" text-anchor="middle">COMPACT DOMAIN K = [0, 1]</text>
		<rect class="viz-node viz-node--output" x="12" y="296" width="336" height="48" rx="4"></rect>
		<text class="viz-callout" x="24" y="316">PROVIDES: SOME WIDTH AND WEIGHTS</text>
		<text class="viz-label" x="24" y="334">Uniform error below ε on the stated compact K.</text>
		<rect class="viz-node viz-node--warning" x="12" y="356" width="336" height="62" rx="4"></rect>
		<text class="viz-callout" x="24" y="376">DOES NOT PROVIDE TRAINING SUCCESS</text>
		<text class="viz-label" x="24" y="395">No generic width or finite-sample generalization bound</text>
		<text class="viz-label" x="24" y="410">No constraint outside K</text>
	</svg>
	<div class="visual-derivation-caption">both curves meet at the three interpolation nodes. Their largest gap occurs between nodes and equals 0.0625, below the 0.07 band. This explicit construction is not a training result.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="uat-existence-error-band"><strong>Check:</strong> What does the pictured error band establish on [0,1]? Which claims about training or generalization remain unsupported?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="uat-existence-error-band">Four straight segments stay close to the curved target throughout this interval. A construction with such a small error demonstrates that a fit exists. It does not establish that training will find it or that it generalizes outside the interval. <strong>Does constructing these four segments prove a training algorithm will find them?</strong></span></figcaption>
</figure>
<p class="diagram-source">The theorem's quantifiers follow <a href="https://hal.science/hal-03753170v1">Cybenko (1989)</a>. The plotted ReLU construction and its error are calculated below.</p>

## Verify the error between nodes

On $[0,0.5]$, $g(x)=0.5x$.
Its error is $x(0.5-x)$, maximized at $x=0.25$ with value $1/16$.
On $[0.5,1]$, $g(x)=1.5x-0.5$.
The error is $(x-0.5)(1-x)$, with the same maximum at $x=0.75$.
Therefore $\sup_K|g-f|=0.0625<0.07$.

At $x=0.25$, $g=0.125$ but $f=0.0625$.
Exact agreement at interpolation nodes did not imply zero error between them.

## What the theorem says (precisely)

One classical version uses a continuous sigmoidal activation $\sigma$.
For every continuous $f$ on compact $K\subset\mathbb R^n$ and every $\varepsilon>0$, some finite-width network

$$
g(x)=\sum_{j=1}^N c_j\sigma(w_j^\mathsf T x+b_j)
$$

satisfies $\sup_{x\in K}|g(x)-f(x)|<\varepsilon$.
The output combines hidden units linearly.
See [Cybenko](https://link.springer.com/article/10.1007/BF02551274) and [Hornik](https://www.sciencedirect.com/science/article/abs/pii/089360809190009T).
ReLU uses separate generalized results, including [Pinkus](https://www.cambridge.org/core/journals/acta-numerica/article/abs/approximation-theory-of-the-mlp-model-in-neural-networks/18072C558C8410C4F92A82BCC8FC8CF9).
An arbitrary nonlinear activation is not sufficient; a fixed-degree polynomial activation does not give this shallow universality.

## Change the approximation requirement

### 1. Require smaller uniform error

**New task:** approximate the same quadratic with error below $0.02$.

**Picture and change:** use four segments of width $h=1/4$:

$$
g_4(x)=0.25\operatorname{ReLU}(x)+0.5\sum_{k=1}^3\operatorname{ReLU}(x-k/4).
$$

**Check:** all five grid values match.
On each interval, interpolation error is at most $h^2/4=1/64=0.015625<0.02$.

**Use when:** the target permits a constructive width-versus-error calculation.

### 2. Introduce a discontinuity

**New task:** uniformly approximate a unit step at $x=0.5$ using continuous network outputs.

**Picture and change:** compare the value at the jump with its left limit.
Continuity forces them to agree, while the target jumps by one.

**Check:** uniform error below $1/2$ would require the same limiting value below and above $1/2$. That is impossible.

**Use when:** checking whether the target-class assumption survives a changed task.

## The bound does not extend outside K

At $x=2$, this $g$ gives $2.5$, while $f$ gives $4$.
The bound on $[0,1]$ says nothing about that point.

## Try interpreting existence

Does existence of suitable weights prove that SGD finds them from random initialization?

<details>
<summary>Check your answer</summary>

No. Optimization and finite-sample generalization need separate assumptions and evidence.
Even a constructive approximation does not establish a generic training procedure.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## What the theorem does **not** say

The existence statement gives no generic small width, successful optimization or statistical generalization bound.
Some proofs are constructive; the missing claim is a generally successful learning procedure.
Without nonlinear activations, stacking affine layers still gives one affine map and cannot approximate arbitrary continuous targets.

<span id="why-deep-nets-are-practically-necessary"></span>

## Depth and representation efficiency

[Telgarsky](https://arxiv.org/abs/1602.04485) and [Eldan and Shamir](https://arxiv.org/abs/1512.03965) give depth-separation results under particular constructions and error measures.
[Mhaskar and Poggio](https://arxiv.org/abs/1603.00988) study compositional structure.
These results are not universal guarantees that deeper models train better or need fewer samples.

## Related theoretical results

- **Barron-type rates:** $O(N^{-1/2})$ approximation in an appropriate $L^2$ norm requires controlled function-class norms; constants matter.
- **Kolmogorov-Arnold:** continuous multivariate functions admit representations through continuous univariate functions. This does not guarantee efficient learned KAN approximations.
- **Bounded-width ReLU:** [Lu et al.](https://arxiv.org/abs/1709.02540) give an $L^1$ universality result for integrable functions with width $n+4$. Do not replace that norm with uniform error.

## What to say in interviews

State target class, domain, activation and error norm.
Distinguish some network existing from the chosen architecture, optimizer and sample achieving the desired outcome.

## Common pitfalls

Matching finitely many samples does not verify uniform error.
Changing the domain, norm or activation requires checking the relevant theorem again.

</div>
