---
title: "Residual connections"
description: "Trace addition through a residual block, including both gradient contributions, projection shortcuts, normalization and cancellation."
date: "2025-11-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A residual block adds a learned change to a shortcut: $y=x+f(x)$.
The identity branch supplies a direct gradient contribution, not a guarantee about the combined gradient.

Take $x=2$, $f(x)=0.1x$ and $L=y^2/2$.
Find the output and the two contributions to $\partial L/\partial x$.

## The mechanism

A residual block:

```
y = x + f(x)
```

The input goes to both branches. Addition joins their values in the forward pass and splits the upstream gradient in backward.

<!-- visual:residual-add-gradient-split -->
<figure class="learning-figure plot-panel" aria-labelledby="residual-gradient-title">
	<p class="visual-kicker">One input, two contributions</p>
	<p class="visual-title" id="residual-gradient-title">Addition copies the upstream gradient; each branch then applies its own derivative</p>
	<svg viewBox="0 0 360 620" role="img" aria-labelledby="residual-gradient-svg-title residual-gradient-svg-desc">
		<title id="residual-gradient-svg-title">Forward and backward paths through one residual addition</title>
		<desc id="residual-gradient-svg-desc">The specified function is f(x)=0.1x, so its derivative is 0.1. Input two splits into f(2)=0.2 and unchanged x=2. Addition gives y=2.2 and loss y squared over two gives upstream gradient 2.2. Backward, the residual branch contributes 0.1 times 2.2, or 0.22. The identity contributes 2.2. Their sum is 2.42.</desc>
		<defs>
			<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
		</defs>
		<text class="viz-axis-label" x="16" y="22">FORWARD · SPLIT, TRANSFORM ONE BRANCH, THEN ADD</text>
		<rect class="viz-node viz-node--input" x="130" y="40" width="100" height="42" rx="5"></rect>
		<text class="viz-node-label" x="180" y="66">input x = 2</text>
		<path class="viz-forward" d="M180 82V105H85V126"></path>
		<path class="viz-forward" d="M180 82V105H275V126"></path>
		<rect class="viz-node" x="25" y="128" width="120" height="54" rx="5"></rect>
		<text class="viz-node-label" x="85" y="151">f(x) = 0.1x</text>
		<text class="viz-node-value" x="85" y="169">f(2) = 0.2</text>
		<rect class="viz-node viz-node--focus" x="215" y="128" width="120" height="54" rx="5"></rect>
		<text class="viz-node-label" x="275" y="151">identity branch</text>
		<text class="viz-node-value" x="275" y="169">x = 2 unchanged</text>
		<path class="viz-forward" d="M85 182V209H161"></path>
		<path class="viz-forward" d="M275 182V209H199"></path>
		<circle class="viz-node viz-node--focus" cx="180" cy="209" r="19"></circle>
		<text class="viz-node-label" x="180" y="215">+</text>
		<path class="viz-forward" d="M180 228V248"></path>
		<rect class="viz-node viz-node--output" x="90" y="250" width="180" height="44" rx="5"></rect>
		<text class="viz-node-label" x="180" y="277">y = 2 + 0.2 = 2.2</text>
		<line class="viz-gridline" x1="16" y1="322" x2="344" y2="322"></line>
		<text class="viz-axis-label" x="16" y="349">BACKWARD · THE ADD NODE SENDS g DOWN BOTH BRANCHES</text>
		<rect class="viz-node viz-node--input" x="80" y="368" width="200" height="44" rx="5"></rect>
		<text class="viz-node-label" x="180" y="387">Loss L = y² / 2</text>
		<text class="viz-node-value" x="180" y="403">g = ∂L/∂y = 2.2</text>
		<path class="viz-backward" d="M180 412V435H85V454"></path>
		<path class="viz-backward" d="M180 412V435H275V454"></path>
		<rect class="viz-node" x="20" y="456" width="130" height="62" rx="5"></rect>
		<text class="viz-node-label" x="85" y="478">through f</text>
		<text class="viz-node-value" x="85" y="496">g · J<tspan baseline-shift="sub" font-size="8">f</tspan> = 0.22</text>
		<text class="viz-node-value" x="85" y="511">f'(x) = 0.1</text>
		<rect class="viz-node viz-node--focus" x="210" y="456" width="130" height="62" rx="5"></rect>
		<text class="viz-node-label" x="275" y="478">direct route</text>
		<text class="viz-node-value" x="275" y="496">g · I = 2.2</text>
		<text class="viz-node-value" x="275" y="511">no residual operation</text>
		<path class="viz-backward" d="M85 518V541H161"></path>
		<path class="viz-backward" d="M275 518V541H199"></path>
		<circle class="viz-node viz-node--focus" cx="180" cy="541" r="19"></circle>
		<text class="viz-node-label" x="180" y="547">+</text>
		<path class="viz-backward" d="M180 560V574"></path>
		<rect class="viz-node viz-node--output" x="55" y="576" width="250" height="34" rx="5"></rect>
		<text class="viz-node-label" x="180" y="599">∂L/∂x = 2.2 + 0.22 = 2.42</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> the forward branches add to 2.2. Backward copies the upstream 2.2 to both branches. The residual branch scales it by 0.1; the shortcut leaves it unchanged. Their contributions add to 2.42. <span class="visual-reader-check" data-reader-check="residual-add-gradient-split"><strong>Check:</strong> Why does the add node copy 2.2 into both branches? Why does the residual branch return only 0.22?</span></figcaption>
</figure>
<p class="diagram-source">References: <a href="https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">Deep Residual Learning</a> and <a href="https://arxiv.org/abs/1603.05027">Identity Mappings in Deep Residual Networks</a>.</p>

The loss is $2.2^2/2=2.42$, giving upstream gradient $g_y=2.2$.
The shortcut contributes $2.2$; the residual branch contributes $0.22$.
Directly differentiating $L=(1.1x)^2/2$ gives $1.21x=2.42$ at $x=2$ too.

For vector inputs, the diagram's row-gradient convention gives

$$
g_x=g_y(I+J_f).
$$

One contribution crosses no residual operation. The sum still depends on $J_f$.

## Change the shortcut

### 1. Change output dimension

**New task:** map $x=(1,2)^\mathsf T$ into three channels with $f=0$.

**Picture and change:** replace the identity wire by
$P=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$, giving $y=Px+f(x)$.
The branches must produce matching shapes before addition.

**Check:** $y=(1,2,3)^\mathsf T$.
Row upstream gradient $(1,1,1)$ returns $g_yP=(2,2)$, not the unchanged upstream vector.

**Use when:** channels or spatial resolution change across a block.

### 2. Move normalization outside the addition

**New task:** compare pre-norm $x+f(\operatorname{LN}(x))$ and post-norm $\operatorname{LN}(x+f(x))$, with $f=0$.
LayerNorm has unit gain and zero offset.

**Picture and change:** add a common shift $c(1,1)$ to $x$.
Pre-norm retains it through the shortcut; post-norm removes it by centering.

**Check:** the directional derivative along $(1,1)$ is $(1,1)$ for pre-norm and zero for post-norm.

**Use when:** reasoning about gradient paths across normalization boundaries.

## The two contributions can cancel

Set $f(x)=-x$. Then $y=0$ and $I+J_f=0$.
The total input gradient is zero even though the identity contribution exists.
An identity branch does not prove a nonvanishing total gradient.

## Try repeating the block

Does repeatedly applying $f(x)=0.1x$ prevent gradient explosion?

<details>
<summary>Check your answer</summary>

No. Each block has local gain $1.1$.
Across $N$ identical scalar blocks, the derivative is $1.1^N$.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Why it works (intuitions)

The [ResNet paper](https://arxiv.org/abs/1512.03385) studied degradation in deeper plain networks, including worse training error.
An additive block can represent identity by setting its residual function to zero.
[Veit et al.](https://arxiv.org/abs/1605.06431) analyze paths of differing depths; this is not an ensemble of independently trained models.

## Pre-norm vs. post-norm in transformers

Their Jacobians are $I+J_fJ_{\mathrm{LN}}$ for pre-norm and $J_{\mathrm{LN}}(I+J_f)$ for post-norm.
Matrix order matters. Neither expression guarantees bounded gradients through a whole network.
See [Xiong et al.](https://arxiv.org/abs/2002.04745) for normalization placement under stated training and initialization assumptions.

## Bottleneck blocks

A bottleneck reduces channels before the expensive spatial convolution, then expands them:

```
y = x + Conv1x1 ↓ → Conv3x3 → Conv1x1 ↑ (x)
```

Ignoring biases, its convolution parameters total $C_{\mathrm{in}}C_m+9C_m^2+C_mC_{\mathrm{out}}$.
Savings depend on widths, spatial sizes and the comparison block, not a fixed factor or accuracy guarantee.

## Where residuals show up

| Architecture | Where |
|--------------|-------|
| ResNet, ResNeXt, Wide ResNet | Every block |
| U-Net | Long skips may concatenate; distinguish those from additive residual blocks |
| Transformer (encoder & decoder) | Around attention and FFN sub-blocks |
| Diffusion U-Nets | Both within blocks and across encoder-decoder |
| MLP-Mixer, ConvNeXt | Block residuals |

## Common pitfalls

Some deep-network recipes scale residual contributions with depth; the scale must match their initialization and training assumptions.
Addition, saved activations and projection shortcuts have memory or compute costs.

## Related

- [Exploding and vanishing gradients](/concepts/exploding-vanishing-gradients/).
- [Transformer architecture](/concepts/transformer-architecture/).

</div>
