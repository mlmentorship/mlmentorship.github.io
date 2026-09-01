---
title: "Residual connections"
description: "Add the input of a block to its output. Lets gradients flow unimpeded through depth and made networks deeper than 30 layers practical for the first time."
date: "2025-11-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **residual connection** (skip connection) makes a block compute $y = x + f(x)$ instead of $y = f(x)$, so that the block's output adds to its input rather than replacing it. Introduced by ResNet [(He et al., 2015)](https://arxiv.org/abs/1512.03385) and ubiquitous in every modern deep architecture.

Pre-ResNet (2014), networks past ~20 layers showed *worse* training accuracy than shallower networks. Not from overfitting but from optimization pathology. Residual connections solved this and made 152-layer ResNets routine, then 1000-layer networks (with normalization) feasible. Every modern architecture. ResNets, transformers, U-Nets, diffusion models, MLP-Mixers. Uses residuals.

## The mechanism

A residual block:

```
y = x + f(x)
```

where $f$ is the "residual function". Typically (Conv → BN → ReLU → Conv → BN) for ResNet or (LayerNorm → Attn → Linear) for transformer attention.

The forward pass is trivial. The interesting effect is on gradients:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left( I + \frac{\partial f}{\partial x} \right).
$$

The identity matrix $I$ in the parenthesis is the "residual gradient highway". Gradients flow back through the identity term without being multiplied by Jacobians of $f$. Even if $f$ is poorly conditioned or near-zero gradient, the identity ensures gradient signal reaches earlier layers.

<!-- visual:residual-add-gradient-split -->
<figure class="learning-figure plot-panel" aria-labelledby="residual-gradient-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="residual-gradient-title">Where does the identity term in a residual block's gradient come from?</p>
	<svg viewBox="0 0 360 620" role="img" aria-labelledby="residual-gradient-svg-title residual-gradient-svg-desc">
		<title id="residual-gradient-svg-title">Forward and backward paths through one residual addition</title>
		<desc id="residual-gradient-svg-desc">In the forward panel, input x splits into a residual branch f of x and an unchanged identity branch x; addition combines them into y equals x plus f of x. In the backward panel, upstream gradient g equals partial L over partial y splits at the same addition. The residual branch multiplies g by the Jacobian J sub f. The identity branch multiplies g by I, which leaves g unchanged and crosses no residual operation. Adding the branch contributions gives partial L over partial x equals g times the quantity I plus J sub f.</desc>
		<defs>
			<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
		</defs>
		<text class="viz-axis-label" x="16" y="22">FORWARD · SPLIT, TRANSFORM ONE BRANCH, THEN ADD</text>
		<rect class="viz-node viz-node--input" x="130" y="40" width="100" height="42" rx="5"></rect>
		<text class="viz-node-label" x="180" y="66">input x</text>
		<path class="viz-forward" d="M180 82V105H85V126"></path>
		<path class="viz-forward" d="M180 82V105H275V126"></path>
		<rect class="viz-node" x="25" y="128" width="120" height="54" rx="5"></rect>
		<text class="viz-node-label" x="85" y="151">residual branch</text>
		<text class="viz-node-value" x="85" y="169">f(x)</text>
		<rect class="viz-node viz-node--focus" x="215" y="128" width="120" height="54" rx="5"></rect>
		<text class="viz-node-label" x="275" y="151">identity branch</text>
		<text class="viz-node-value" x="275" y="169">x unchanged</text>
		<path class="viz-forward" d="M85 182V209H161"></path>
		<path class="viz-forward" d="M275 182V209H199"></path>
		<circle class="viz-node viz-node--focus" cx="180" cy="209" r="19"></circle>
		<text class="viz-node-label" x="180" y="215">+</text>
		<path class="viz-forward" d="M180 228V248"></path>
		<rect class="viz-node viz-node--output" x="90" y="250" width="180" height="44" rx="5"></rect>
		<text class="viz-node-label" x="180" y="277">y = x + f(x)</text>
		<line class="viz-gridline" x1="16" y1="322" x2="344" y2="322"></line>
		<text class="viz-axis-label" x="16" y="349">BACKWARD · THE ADD NODE SENDS g DOWN BOTH BRANCHES</text>
		<rect class="viz-node viz-node--input" x="80" y="368" width="200" height="44" rx="5"></rect>
		<text class="viz-node-label" x="180" y="387">upstream gradient</text>
		<text class="viz-node-value" x="180" y="403">g = ∂L/∂y</text>
		<path class="viz-backward" d="M180 412V435H85V454"></path>
		<path class="viz-backward" d="M180 412V435H275V454"></path>
		<rect class="viz-node" x="20" y="456" width="130" height="62" rx="5"></rect>
		<text class="viz-node-label" x="85" y="478">through f</text>
		<text class="viz-node-value" x="85" y="496">g · J<tspan baseline-shift="sub" font-size="8">f</tspan></text>
		<text class="viz-node-value" x="85" y="511">uses residual Jacobian</text>
		<rect class="viz-node viz-node--focus" x="210" y="456" width="130" height="62" rx="5"></rect>
		<text class="viz-node-label" x="275" y="478">direct route</text>
		<text class="viz-node-value" x="275" y="496">g · I = g</text>
		<text class="viz-node-value" x="275" y="511">no residual operation</text>
		<path class="viz-backward" d="M85 518V541H161"></path>
		<path class="viz-backward" d="M275 518V541H199"></path>
		<circle class="viz-node viz-node--focus" cx="180" cy="541" r="19"></circle>
		<text class="viz-node-label" x="180" y="547">+</text>
		<path class="viz-backward" d="M180 560V574"></path>
		<rect class="viz-node viz-node--output" x="55" y="576" width="250" height="34" rx="5"></rect>
		<text class="viz-node-label" x="180" y="599">∂L/∂x = g · (I + J<tspan baseline-shift="sub" font-size="9">f</tspan>)</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> start with the forward split: only the left branch computes f(x), while the right branch copies x to the add node. During backward, that add node copies the same upstream gradient g to both inputs. The left contribution becomes g · J<sub>f</sub>; the direct contribution stays g · I = g. Adding them produces g · (I + J<sub>f</sub>), so one term reaches x without crossing f. Original schematic checked against <a href="https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">Deep Residual Learning</a> and the derivation in <a href="https://arxiv.org/abs/1603.05027">Identity Mappings in Deep Residual Networks</a>.</figcaption>
</figure>

## Why it works (intuitions)

Three complementary explanations:

1. **Easier to learn the identity.** If the optimal $f$ is near zero (i.e., the layer is unhelpful), the network can simply set $f \to 0$ and the block becomes the identity. Without the residual, learning the identity through a deep stack of ReLU+linear is hard.
2. **Gradient highway.** As above; identity term in the backward pass prevents vanishing.
3. **Implicit ensemble** [(Veit et al., 2016)](https://arxiv.org/abs/1605.06431): a depth-$N$ ResNet acts like an ensemble of $2^N$ paths of varying depth, with the shallow paths providing strong learning signal.

## Pre-norm vs. post-norm in transformers

Two arrangements of the residual + normalization in transformer blocks:

- **Post-norm** (original transformer): $y = \text{LayerNorm}(x + f(x))$. Used in original Vaswani et al. (2017).
- **Pre-norm**: $y = x + f(\text{LayerNorm}(x))$. Used in GPT-2/3, Llama, Mistral, every modern decoder.

Pre-norm is much more stable to train at depth; the residual stream is never normalized, so gradient magnitudes stay bounded. Post-norm requires careful warmup. Pre-norm is the default in 2026.

## Bottleneck blocks

For very deep networks (ResNet-50/101/152), the residual block is replaced with a **bottleneck**:

```
y = x + Conv1x1 ↓ → Conv3x3 → Conv1x1 ↑ (x)
```

Reduce channels with $1 \times 1$ conv, do the expensive $3 \times 3$ at low channel count, expand back. Cuts compute roughly 4× per block at similar accuracy.

## Where residuals show up

| Architecture | Where |
|--------------|-------|
| ResNet, ResNeXt, Wide ResNet | Every block |
| U-Net | Across encoder-decoder paths (long skips) |
| Transformer (encoder & decoder) | Around attention and FFN sub-blocks |
| Diffusion U-Nets | Both within blocks and across encoder-decoder |
| MLP-Mixer, ConvNeXt | Block residuals |

Almost no modern architecture omits residuals.

## Common pitfalls

- **Adding residual through dimension change.** $x + f(x)$ requires matching shapes. When channel count changes, project $x$ with a $1 \times 1$ conv (ResNet) or linear (transformer with embedding-dim mismatch).
- **Putting normalization before vs. after the residual.** Pre-norm vs. post-norm have very different training dynamics; pre-norm is the safe choice in 2026.
- **Skipping the residual scaling in deep stacks.** Some recipes scale the residual contribution by $1/\sqrt{N}$ for $N$ layers (GPT-2 style); useful for very deep stacks.
- **Treating residuals as "free."** They cost a small amount of memory (need to keep $x$ around for the addition) and contribute to activation memory.

## Related

- [Exploding and vanishing gradients](/concepts/exploding-vanishing-gradients/). The problem residuals solve.
- [Transformer architecture](/concepts/transformer-architecture/). The canonical user.
