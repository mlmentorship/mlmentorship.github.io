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
	<p class="visual-title" id="activation-regions-title">Flat activation regions weaken the gradient travelling backward</p><div class="visual-argument" data-visual-argument="activation-gradient-regions" data-argument-kind="curve" data-argument-composition="local-slope"><svg viewBox="0 0 360 594" role="img" aria-label="A flat forward response has little backward slope. ReLU has zero derivative on its negative side. Sigmoid slopes become small in both tails; their forward outputs are not gradients."><title>A flat forward response has little backward slope</title><desc>ReLU has zero derivative on its negative side. Sigmoid slopes become small in both tails; their forward outputs are not gradients.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A flat forward response gives a small</tspan><tspan x="20" dy="18">slope</tspan></text><text x="20" y="79" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Sigmoid: output zero to one</tspan></text><path d="M35 218L325 218" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M35 216.02151690416994L38.625 215.81756636925147L42.25000000000001 215.59306019702564L45.87499999999999 215.3460276440564L49.5 215.07433070654474L53.125 214.7756546173508L56.75 214.44749888317045L60.37500000000001 214.08716918001002L63.99999999999999 213.69177049235591L67.625 213.25820195648052L71.25 212.78315395046766L74.875 212.26310806137406L78.5 211.69434065112443L82.125 211.0729308337304L85.75 210.39477376223186L89.375 209.6556001976632L93 208.85100338566855L96.625 207.9764742883658L100.25 207.02744619683463L103.875 205.99934966848258L107.5 204.88767857756707L111.125 203.68806782007024L114.75 202.39638286094635L118.375 201.00882084081118L122 199.5220223647317L125.625 197.9331923813008L129.25 196.240227741444L132.875 194.44184813468144L136.5 192.53772618489194L140.125 190.52861161546295L143.75 188.41644364930053L147.375 186.20444528875043L151 183.89719292403737L154.625 181.5006549384983L158.25 179.0221936848375L161.875 176.470526432204L165.5 173.8556426123697L169.125 171.18867684928247L172.75 168.48173970437256L176.375 165.7477106226834L180 163L183.625 160.2522893773166L187.25 157.51826029562744L190.875 154.81132315071753L194.5 152.14435738763027L198.125 149.529473567796L201.75 146.97780631516252L205.375 144.49934506150174L209 142.10280707596263L212.625 139.79555471124957L216.25 137.58355635069947L219.875 135.47138838453708L223.5 133.46227381510806L227.125 131.55815186531856L230.75 129.75977225855598L234.375 128.06680761869922L238 126.4779776352683L241.625 124.99117915918882L245.25 123.60361713905365L248.875 122.31193217992977L252.5 121.11232142243294L256.125 120.00065033151742L259.75 118.97255380316537L263.375 118.02352571163418L267 117.14899661433145L270.625 116.34439980233678L274.25 115.60522623776816L277.875 114.92706916626962L281.5 114.30565934887557L285.125 113.73689193862594L288.75 113.21684604953234L292.375 112.74179804351948L296 112.30822950764407L299.625 111.91283081998998L303.25 111.55250111682955L306.875 111.2243453826492L310.5 110.92566929345526L314.125 110.65397235594361L317.75 110.40693980297436L321.375 110.18243363074852L325 109.97848309583007" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2"></path><path d="M56.75 214.77092697862778L85.75 210.79538092230752" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><circle cx="71.25" cy="212.78315395046766" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="71.25" y="254" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="71.25" dy="0">0.04518</tspan></text><path d="M165.5 174L194.5 152" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><circle cx="180" cy="163" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="180" y="254" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="180" dy="0">0.25</tspan></text><path d="M274.25 115.20461907769246L303.25 111.2290730213722" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><circle cx="288.75" cy="113.21684604953234" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="288.75" y="254" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="288.75" dy="0">0.04518</tspan></text><text x="20" y="301" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">ReLU: output zero to four</tspan></text><path d="M35 462L325 462" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M35 462L38.625 462L42.25000000000001 462L45.87499999999999 462L49.5 462L53.125 462L56.75 462L60.37500000000001 462L63.99999999999999 462L67.625 462L71.25 462L74.875 462L78.5 462L82.125 462L85.75 462L89.375 462L93 462L96.625 462L100.25 462L103.875 462L107.5 462L111.125 462L114.75 462L118.375 462L122 462L125.625 462L129.25 462L132.875 462L136.5 462L140.125 462L143.75 462L147.375 462L151 462L154.625 462L158.25 462L161.875 462L165.5 462L169.125 462L172.75 462L176.375 462L180 462L183.625 458.8L187.25 455.6L190.875 452.4L194.5 449.2L198.125 446L201.75 442.8L205.375 439.6L209 436.4L212.625 433.2L216.25 430L219.875 426.8L223.5 423.6L227.125 420.4L230.75 417.2L234.375 414L238 410.8L241.625 407.6L245.25 404.4L248.875 401.2L252.5 398L256.125 394.8L259.75 391.6L263.375 388.4L267 385.2L270.625 382L274.25 378.8L277.875 375.6L281.5 372.4L285.125 369.2L288.75 366L292.375 362.8L296 359.6L299.625 356.4L303.25 353.2L306.875 350L310.5 346.8L314.125 343.6L317.75 340.4L321.375 337.2L325 334" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path><path d="M93 462L122 462" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><circle cx="107.5" cy="462" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="107.5" y="498" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="107.5" dy="0">0</tspan></text><path d="M238 410.8L267 385.2" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><circle cx="252.5" cy="398" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="252.5" y="498" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="252.5" dy="0">1</tspan></text><text x="20" y="550" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Numbers are local slopes, not forward</tspan><tspan x="20" dy="18">outputs.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> sigmoid loses slope in both tails. ReLU keeps unit slope for positive inputs and zero slope for negative inputs. Exact GELU has a smooth negative shoulder, but its derivative can still equal zero.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="activation-gradient-regions"><strong>Check:</strong> Where would an incoming gradient shrink or become zero? How is that different for GELU?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="activation-gradient-regions">ReLU has zero derivative on its negative side. Sigmoid slopes become small in both tails; their forward outputs are not gradients. <strong>Where does the same small input move produce almost no output change?</strong></span></figcaption>
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
