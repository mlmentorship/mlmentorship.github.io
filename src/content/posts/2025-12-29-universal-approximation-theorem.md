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
	<p class="visual-title" id="uat-error-band-title">An explicit approximation inside the band proves existence here, not trainability or generalization</p>
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
	<figcaption><strong>Read it this way:</strong> both curves meet at the three interpolation nodes. Their largest gap occurs between nodes and equals 0.0625, below the 0.07 band. This explicit construction is not a training result. <span class="visual-reader-check" data-reader-check="uat-existence-error-band"><strong>Check:</strong> What does the pictured error band establish on [0,1]? Which claims about training or generalization remain unsupported?</span></figcaption>
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
