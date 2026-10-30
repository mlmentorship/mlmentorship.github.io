---
title: "Activation functions"
description: "Trace how activations introduce breakpoints, change gradients and constrain outputs. Compare ReLU, GELU, sigmoid and gated projections."
date: "2026-01-19"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An activation inserts a nonlinear map between affine layers.
Without it, repeated affine layers collapse into one affine map.

Take $h=2x+1$, then compare $y=3h-4$ with $y=3\operatorname{ReLU}(h)-4$.
What changes at $x=-1$ and $x=1$?
The graphs show activation output $g(h)$, not the complete network output $y$.

<!-- visual:activation-gradient-regions -->
<figure class="learning-figure" aria-labelledby="activation-regions-title">
	<p class="visual-kicker">Optimization intuition</p>
	<p class="visual-title" id="activation-regions-title">Flat activation regions weaken the gradient travelling backward</p><div class="visual-argument" data-visual-argument="activation-gradient-regions" data-argument-kind="curve"><p class="visual-argument-premise">A flat forward response has little backward slope</p><svg viewBox="0 0 360 375" role="img" aria-label="A flat forward response has little backward slope. ReLU has zero derivative on its negative side. Sigmoid slopes become small in both tails; their forward outputs are not gradients."><title>A flat forward response has little backward slope</title><desc>ReLU has zero derivative on its negative side. Sigmoid slopes become small in both tails; their forward outputs are not gradients.</desc><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">-5</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">5</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">1</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-series="Sigmoid derivative" d="M48 218.86983036596567L52.40625 218.68168388580725L56.8125 218.46279877765832L61.21875 218.20836407506894L65.625 217.91289216525746L70.03125 217.570151602522L74.4375 217.17310417409604L78.84375 216.71385170401444L83.25 216.1836002350526L87.65625 215.57265192869644L92.0625 214.8704382997521L96.46875 214.06561221892082L100.875 213.14622031733165L105.28125 212.09998166465215L109.6875 210.9147022375013L114.09375 209.57885673062347L118.5 208.0823681873316L122.90625 206.4176096997234L127.3125 204.5806385449556L131.71875 202.57264888708156L136.125 200.40159232550465L140.53125 198.0838653479831L144.9375 195.64590138542928L149.34375 193.12543963556752L153.75 190.5721862112755L158.15625 188.0475547553345L162.5625 185.62319643878166L166.96875 183.37812396903345L171.375 181.39441015987353L175.78125 179.7516864628388L180.1875 178.520940376298L184.59375 177.7583475541814L189 177.5L193.40625 177.7583475541814L197.8125 178.520940376298L202.21875 179.7516864628388L206.625 181.39441015987353L211.03125 183.37812396903345L215.4375 185.62319643878166L219.84375 188.0475547553345L224.25 190.5721862112755L228.65625 193.12543963556752L233.0625 195.64590138542928L237.46875 198.0838653479831L241.875 200.40159232550462L246.28125 202.57264888708153L250.6875 204.58063854495563L255.09375 206.41760969972344L259.5 208.08236818733164L263.90625 209.57885673062344L268.3125 210.91470223750127L272.71875 212.09998166465215L277.125 213.14622031733165L281.53125 214.06561221892082L285.9375 214.8704382997521L290.34375 215.57265192869644L294.75 216.1836002350526L299.15625 216.71385170401444L303.5625 217.17310417409604L307.96875 217.57015160252197L312.375 217.91289216525743L316.78125 218.20836407506894L321.1875 218.4627987776583L325.59375 218.68168388580725L330 218.8698303659657" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></path><path data-proof-series="ReLU derivative (except at zero)" d="M48 220L188.97179999999997 220L189.0282 50L330 50" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2.5;stroke-dasharray:6 3"></path><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">input</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">local derivative</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Sigmoid derivative</tspan></text><path d="M20 336L48 336" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="340" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">ReLU derivative (except at zero)</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two">
		<section class="visual-panel plot-panel" aria-labelledby="sigmoid-panel-heading">
			<h4 id="sigmoid-panel-heading">Sigmoid: two saturating tails</h4>
			<p>The slope approaches zero at both output limits.</p>
			<svg viewBox="0 0 320 230" role="img" aria-labelledby="sigmoid-svg-title sigmoid-svg-desc">
				<title id="sigmoid-svg-title">Logistic sigmoid from negative four to positive four</title>
				<desc id="sigmoid-svg-desc">The sigmoid rises smoothly through one half at input zero and flattens toward output zero on the left and output one on the right. Both flat tails are labeled as saturation regions where the derivative approaches zero.</desc>
				<rect class="viz-plot-bg" x="42" y="24" width="250" height="166" rx="3"></rect>
				<path class="viz-gridline" d="M42 123H292 M42 57H292"></path>
				<path class="viz-axis" d="M167 24V190 M42 190H292"></path>
				<path class="viz-roc-curve" d="M42 187.6 L49.8 186.9 L57.6 186.1 L65.4 185 L73.3 183.7 L81.1 182 L88.9 179.9 L96.7 177.3 L104.5 174.1 L112.3 170.3 L120.1 165.7 L127.9 160.3 L135.8 154.1 L143.6 147.2 L151.4 139.7 L159.2 131.6 L167 123.3 L174.8 115 L182.6 107 L190.4 99.4 L198.3 92.5 L206.1 86.4 L213.9 81 L221.7 76.4 L229.5 72.6 L237.3 69.4 L245.1 66.8 L252.9 64.7 L260.8 63 L268.6 61.6 L276.4 60.6 L284.2 59.7 L292 59.1"></path>
				<text class="viz-callout" x="48" y="130">left tail:</text>
				<text class="viz-callout" x="48" y="144">σ′(h) → 0</text>
				<text class="viz-callout" x="185" y="46">saturated: σ′(h) → 0</text>
				<circle class="viz-operating-point" cx="167" cy="123" r="4"></circle>
				<text class="viz-label" x="173" y="119">σ(0) = 0.5</text>
				<text class="viz-label" x="38" y="207">−4</text><text class="viz-label" x="163" y="207">0</text><text class="viz-label" x="288" y="207">4</text>
				<text class="viz-label" x="25" y="193">0</text><text class="viz-label" x="17" y="127">0.5</text><text class="viz-label" x="25" y="61">1</text>
				<text class="viz-axis-label" x="281" y="222">input h</text>
				<text class="viz-axis-label" x="46" y="20">output</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel" aria-labelledby="rectifier-panel-heading">
			<h4 id="rectifier-panel-heading">ReLU vs exact GELU</h4>
			<p>A half-line of zero slope differs from one smooth shoulder.</p>
			<svg viewBox="0 0 320 230" role="img" aria-labelledby="rectifier-svg-title rectifier-svg-desc">
				<title id="rectifier-svg-title">ReLU and exact GELU from negative three to positive three</title>
				<desc id="rectifier-svg-desc">ReLU is exactly zero with zero derivative for every negative input, then follows a straight line after zero. Exact GELU, x times the standard normal cumulative distribution, has small negative outputs, a shallow minimum near negative three quarters, passes through zero, and approaches the same positive linear trend. ReLU is solid and GELU is dashed.</desc>
				<rect class="viz-plot-bg" x="42" y="24" width="250" height="166" rx="3"></rect>
				<path class="viz-gridline" d="M42 125H292 M42 82H292 M42 39H292"></path>
				<path class="viz-axis" d="M167 24V190 M42 168.4H292"></path>
				<path class="viz-roc-curve" d="M42 168.4 L83.7 168.4 L125.3 168.4 L167 168.4 L208.7 125.1 L250.3 81.9 L292 38.6"></path>
				<path class="viz-pr-curve" stroke-dasharray="6 4" d="M42 168.6 L47.2 168.6 L52.4 168.7 L57.6 168.9 L62.8 169 L68 169.3 L73.3 169.6 L78.5 169.9 L83.7 170.3 L88.9 170.8 L94.1 171.4 L99.3 172 L104.5 172.7 L109.7 173.4 L114.9 174.1 L120.1 174.7 L125.3 175.2 L130.5 175.6 L135.8 175.7 L141 175.6 L146.2 175 L151.4 174.1 L156.6 172.7 L161.8 170.8 L167 168.4 L172.2 165.4 L177.4 161.9 L182.6 157.9 L187.8 153.4 L193 148.5 L198.3 143.3 L203.5 137.8 L208.7 132 L213.9 126.1 L219.1 120 L224.3 113.9 L229.5 107.8 L234.7 101.8 L239.9 95.7 L245.1 89.8 L250.3 83.9 L255.5 78 L260.8 72.3 L266 66.6 L271.2 60.9 L276.4 55.4 L281.6 49.8 L286.8 44.3 L292 38.8"></path>
				<text class="viz-callout" x="49" y="155">ReLU: inactive, g′(h) = 0</text>
				<text class="viz-callout" x="203" y="112">ReLU: solid</text>
				<text class="viz-callout" x="180" y="137">GELU: dashed</text>
				<text class="viz-label" x="75" y="125">smooth negative shoulder</text>
				<circle class="viz-operating-point" cx="167" cy="168.4" r="4"></circle>
				<text class="viz-label" x="172" y="181">both pass through (0, 0)</text>
				<text class="viz-label" x="38" y="207">−3</text><text class="viz-label" x="163" y="207">0</text><text class="viz-label" x="288" y="207">3</text>
				<text class="viz-label" x="25" y="172">0</text><text class="viz-label" x="25" y="129">1</text><text class="viz-label" x="25" y="86">2</text><text class="viz-label" x="25" y="43">3</text>
				<text class="viz-axis-label" x="281" y="222">input h</text>
				<text class="viz-axis-label" x="46" y="20">output</text>
			</svg>
		</section>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> sigmoid loses slope in both tails. ReLU keeps unit slope for positive inputs and zero slope for negative inputs. Exact GELU has a smooth negative shoulder, but its derivative can still equal zero.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="activation-gradient-regions"><strong>Check:</strong> Where would an incoming gradient shrink or become zero? How is that different for GELU?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="activation-gradient-regions">ReLU has zero derivative on its negative side. Sigmoid slopes become small in both tails; their forward outputs are not gradients.</span></figcaption>
</figure>

## Trace the two inputs

Without activation, substitution gives $y=3(2x+1)-4=6x-1$ everywhere.
ReLU instead clips negative $h$ to zero.

| Input $x$ | $h=2x+1$ | Affine output | ReLU output | ReLU path derivative $dy/dx$ |
| --- | --- | --- | --- | --- |
| $-1$ | $-1$ | $-7$ | $-4$ | $0$ |
| $1$ | $3$ | $5$ | $5$ | $6$ |

The activated network is constant at $-4$ when $x<-1/2$, and equals $6x-1$ when $x>-1/2$.
Its breakpoint is $x=-1/2$. Both branches meet there, but the derivative is not unique.
Frameworks choose a derivative convention at the kink.

For positive $h$, ReLU's local slope is one.
For negative $h$, it is zero, removing this path's gradient on that input.
Sigmoid has derivative $\sigma(h)(1-\sigma(h))$, which approaches zero in both tails.
Full gradient flow also depends on weights and other paths.

## Change the requirement

### 1. Keep a negative-region gradient

**New task:** this path repeatedly receives $h=-2$ and needs a negative-region update.

**Picture and change:** replace ReLU by leaky ReLU with negative slope $0.1$.
This changes both values and gradients.

**Check:** activation output becomes $-0.2$ with local slope $0.1$, instead of zero output and slope.
The complete output is $-4.6$, and $dy/dx=3(0.1)(2)=0.6$.

**Use when:** persistent inactivity is observed, with the changed function acceptable for the task.

### 2. Change label exclusivity

**New task:** logits are $(0,\log4)$.
Choose outputs for one exclusive class versus two labels that can coexist.

**Picture and change:** softmax normalizes jointly; separate sigmoids produce individual Bernoulli probabilities.

**Check:** softmax gives $(0.2,0.8)$; sigmoids give $(0.5,0.8)$.
The latter need not sum to one.

**Use when:** target structure changes, rather than merely changing a hidden activation.

## Check the loss input contract

Do not apply softmax before a cross-entropy API that expects logits.
It would transform already transformed values.
Check whether the API accepts logits, probabilities or log probabilities.

## Try diagnosing a negative ReLU input

Does one negative input mean the unit is permanently dead?

<details>
<summary>Check your answer</summary>

No. Other inputs or upstream parameter changes can reactivate it.
Persistent inactivity across relevant inputs is the concern.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## The standard family

Here $\alpha>0$ for ELU and $0<\alpha<1$ for leaky ReLU.

| Activation | Formula | Range | Constraint or role |
| --- | --- | --- | --- |
| Sigmoid | $\sigma(h)=1/(1+e^{-h})$ | $(0,1)$ | Bernoulli outputs and gates; saturating tails |
| Tanh | $\tanh h=2\sigma(2h)-1$ | $(-1,1)$ | Odd and centered at the origin, not necessarily zero-mean over data |
| ReLU | $\max(0,h)$ | $[0,\infty)$ | Unit positive slope, zero negative slope |
| Leaky ReLU | $\max(\alpha h,h)$ | $\mathbb R$ | Nonzero negative slope |
| ELU | $h$ if $h>0$; $\alpha(e^h-1)$ otherwise | $(-\alpha,\infty)$ | Negative outputs; smooth at zero only when $\alpha=1$ |
| Exact GELU | $h\Phi(h)$, normal CDF $\Phi$ | Bounded below, unbounded above | Small negative shoulder; not every real output is attainable |
| SiLU / swish with unit scale | $h\sigma(h)$ | Bounded below, unbounded above | Smooth, nonmonotone negative shoulder |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | Probability simplex interior for finite logits | Couples coordinates; not an elementwise activation |

<span id="why-relu-won-then-why-geluswish-replaced-it"></span>

## Smoothness does not guarantee gradient flow

Exact GELU has derivative $\Phi(h)+h\phi(h)$, where $\phi$ is the normal density.
It can vanish at isolated points.
Exact and approximate GELU differ; match framework settings for reproducibility.
Activation choice and initialization must be considered together.
No activation guarantees better task accuracy from its curve alone.

References: [ReLU](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf), [GELU](https://arxiv.org/abs/1606.08415), [swish](https://arxiv.org/abs/1710.05941), and [feedforward networks](https://www.deeplearningbook.org/contents/mlp.html).

## SwiGLU and gated activations

For row-vector batches $x\in\mathbb R^{n\times d}$, use $W_g,W_u\in\mathbb R^{d\times m}$ and $W_d\in\mathbb R^{m\times d}$:

$$
\operatorname{SwiGLU}(x)=(\operatorname{SiLU}(xW_g)\odot xW_u)W_d.
$$

This has $3dm$ projection parameters, excluding biases, versus $2dm$ for a two-projection block at equal hidden width.
Equal parameter budgets require adjusting width; performance comparisons must state that budget.

## When to use which output activation

| Task | Output activation |
|------|------------------|
| Binary classification | Sigmoid |
| Multi-class classification | Softmax |
| Multi-label classification | Separate sigmoid heads; labels need not be mutually exclusive |
| Regression (unbounded) | Identity (no activation) |
| Regression (bounded $[0, 1]$) | Sigmoid |
| Probability over a discrete distribution | Softmax |
| Embedding output | Identity; normalize if the similarity objective requires it |

## Common pitfalls

Output constraints must match the target: softplus is positive, while ReLU can produce exact zeros and inactive gradients.
Bounded sigmoid outputs do not by themselves establish calibrated probabilities.

## Related

- [Weight initialization](/concepts/weight-initialization/). Initialization is activation-dependent.
- [Exploding and vanishing gradients](/concepts/exploding-vanishing-gradients/). What motivates ReLU and gated activations.

</div>
