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
	<p class="visual-title" id="uat-error-band-title">An explicit approximation inside the band proves existence here, not trainability or generalization</p><div class="visual-argument" data-visual-argument="uat-existence-error-band" data-argument-kind="curve"><p class="visual-argument-premise">An explicit approximation stays inside an error ceiling</p><svg viewBox="0 0 360 393" role="img" aria-label="An explicit approximation stays inside an error ceiling. Four linear segments approximate x squared with maximum error 1/64, below 0.02. This demonstrates approximation here, not trainability or generalization."><title>An explicit approximation stays inside an error ceiling</title><desc>Four linear segments approximate x squared with maximum error 1/64, below 0.02. This demonstrates approximation here, not trainability or generalization.</desc><path d="M55 40L55 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M55 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="55" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="55" dy="0">0</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">1</tspan></text><text x="47" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="47" dy="0">0.022</tspan></text><text x="47" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="47" dy="0">0</tspan></text><path data-proof-series="Error of four-segment interpolant" d="M55 220L59.296875 191.70188210227272L63.59375 167.1768465909091L67.890625 146.4248934659091L72.1875 129.44602272727272L76.484375 116.240234375L80.78125 106.8075284090909L85.078125 101.14790482954544L89.375 99.26136363636363L93.671875 101.14790482954544L97.96875 106.8075284090909L102.265625 116.240234375L106.5625 129.44602272727272L110.859375 146.4248934659091L115.15625 167.1768465909091L119.453125 191.70188210227272L123.75 220L128.046875 191.70188210227272L132.34375 167.1768465909091L136.640625 146.4248934659091L140.9375 129.44602272727272L145.234375 116.240234375L149.53125 106.8075284090909L153.828125 101.14790482954544L158.125 99.26136363636363L162.421875 101.14790482954544L166.71875 106.8075284090909L171.015625 116.240234375L175.3125 129.44602272727272L179.609375 146.4248934659091L183.90625 167.1768465909091L188.203125 191.70188210227272L192.5 220L196.796875 191.70188210227272L201.09375 167.1768465909091L205.390625 146.4248934659091L209.6875 129.44602272727272L213.984375 116.240234375L218.28125 106.8075284090909L222.578125 101.14790482954544L226.875 99.26136363636363L231.171875 101.14790482954544L235.46875 106.8075284090909L239.765625 116.240234375L244.0625 129.44602272727272L248.359375 146.4248934659091L252.65625 167.1768465909091L256.953125 191.70188210227272L261.25 220L265.546875 191.70188210227272L269.84375 167.1768465909091L274.140625 146.4248934659091L278.4375 129.44602272727272L282.734375 116.240234375L287.03125 106.8075284090909L291.328125 101.14790482954544L295.625 99.26136363636363L299.921875 101.14790482954544L304.21875 106.8075284090909L308.515625 116.240234375L312.8125 129.44602272727272L317.109375 146.4248934659091L321.40625 167.1768465909091L325.703125 191.70188210227272L330 220" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></path><path data-proof-series="Required tolerance 0.02" d="M55 65.45454545454544L330 65.45454545454544" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-dasharray:6 3"></path><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">input x</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">absolute approximation error</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Error of four-segment</tspan><tspan x="60" dy="18">interpolant</tspan></text><path d="M20 354L48 354" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="358" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Required tolerance 0.02</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">both curves meet at the three interpolation nodes. Their largest gap occurs between nodes and equals 0.0625, below the 0.07 band. This explicit construction is not a training result.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="uat-existence-error-band"><strong>Check:</strong> What does the pictured error band establish on [0,1]? Which claims about training or generalization remain unsupported?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="uat-existence-error-band">Four linear segments approximate x squared with maximum error 1/64, below 0.02. This demonstrates approximation here, not trainability or generalization.</span></figcaption>
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
