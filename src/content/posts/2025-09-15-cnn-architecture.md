---
title: "CNN architecture"
description: "Calculate convolution shapes, shared parameters, arithmetic, and receptive fields, then distinguish spatial information from channel capacity."
date: "2025-09-15"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A convolution reuses one local linear operator across spatial positions; a CNN composes these operators with nonlinearities and other layers.
Weight sharing ties parameters, while stride, padding, dilation, and channel counts determine shapes and information loss.
The architectural bias does not by itself guarantee invariance, semantic hierarchy, or superior sample efficiency.

## The building block: convolutional layer

Use channels-last notation here, with input $H\times W\times C_{in}$.
An ungrouped layer has $k_Hk_WC_{in}C_{out}$ weights and optionally $C_{out}$ biases.
Along either spatial axis,

$$
H_{out}=\left\lfloor\frac{H+2p-d(k-1)-1}{s}\right\rfloor+1.
$$

Here $p,d,s$ are padding, dilation, and stride; require valid positive output dimensions.

For unit stride and dilation, omitting boundary indexing, the common cross-correlation implementation computes:

$$
y_{ij,c_{\text{out}}} = \sum_{c_{\text{in}}} \sum_{u, v} W_{u, v, c_{\text{in}}, c_{\text{out}}} \cdot x_{i+u, j+v, c_{\text{in}}} + b_{c_{\text{out}}}.
$$

For a numerical fixture, use input $32\times32\times3$, sixteen $3\times3$ filters, stride one, padding one, and dilation one.
Output is $32\times32\times16$; parameters total $3(3)(3)(16)+16=448$.
There are $32(32)(16)(27)=442368$ multiply-accumulates, excluding biases.
Counting one multiply and one add per MAC gives $884736$ FLOPs under that convention.
The diagram uses different illustrative channel counts: thirty-two, then sixty-four.

<!-- visual:cnn-spatial-channel-pyramid -->
<figure class="learning-figure" aria-labelledby="cnn-pyramid-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="cnn-pyramid-title">Track what changes through a CNN: spatial grids shrink, channels grow, and context widens.</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 242" role="img" aria-labelledby="cnn-pyramid-svg-title cnn-pyramid-svg-desc">
			<title id="cnn-pyramid-svg-title">CNN spatial and channel feature pyramid</title>
			<desc id="cnn-pyramid-svg-desc">Four numbered stages run left to right. A 32 by 32 RGB input becomes a 32 by 32 stack of 32 local feature maps, then a downsampled 16 by 16 stack of 64 feature maps, and finally 64 values after global average pooling. The drawn squares become spatially smaller while their offset stacks become deeper. Labels below state that spatial resolution decreases, channel count increases, and receptive field grows.</desc>
			<text class="viz-axis-label" x="49" y="20" text-anchor="middle">1  Input</text>
			<text class="viz-axis-label" x="137" y="20" text-anchor="middle">2  Local features</text>
			<text class="viz-axis-label" x="231" y="20" text-anchor="middle">3  Downsample</text>
			<text class="viz-axis-label" x="319" y="20" text-anchor="middle">4  Pool</text>
			<rect x="23" y="51" width="58" height="58" rx="2" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
			<path d="M42 70H62M42 80H62M42 90H62M42 70V90M52 70V90M62 70V90" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1"></path>
			<path d="M88 80H105M99 74L105 80L99 86" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
			<rect x="117" y="49" width="58" height="58" rx="2" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.4"></rect>
			<rect x="113" y="53" width="58" height="58" rx="2" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.4"></rect>
			<rect x="109" y="57" width="58" height="58" rx="2" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="120" y="66" width="16" height="16" style="fill:none;stroke:var(--c-text);stroke-width:2;stroke-dasharray:3 2"></rect>
			<path d="M181 80H198M192 74L198 80L192 86" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
			<rect x="222" y="53" width="42" height="42" rx="2" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.2"></rect>
			<rect x="218" y="57" width="42" height="42" rx="2" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.2"></rect>
			<rect x="214" y="61" width="42" height="42" rx="2" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.2"></rect>
			<rect x="210" y="65" width="42" height="42" rx="2" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
			<rect x="218" y="73" width="25" height="25" style="fill:none;stroke:var(--c-text);stroke-width:2;stroke-dasharray:3 2"></rect>
			<path d="M271 80H288M282 74L288 80L282 86" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
			<rect x="320" y="57" width="10" height="48" rx="2" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.2"></rect>
			<rect x="316" y="61" width="10" height="48" rx="2" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.2"></rect>
			<rect x="312" y="65" width="10" height="48" rx="2" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect>
			<text class="viz-axis-label" x="52" y="130" text-anchor="middle">32 &#215; 32 &#215; 3</text>
			<text class="viz-axis-label" x="138" y="130" text-anchor="middle">32 &#215; 32 &#215; 32</text>
			<text class="viz-axis-label" x="232" y="130" text-anchor="middle">16 &#215; 16 &#215; 64</text>
			<text class="viz-axis-label" x="321" y="130" text-anchor="middle">1 &#215; 1 &#215; 64</text>
			<text class="viz-label" x="138" y="146" text-anchor="middle">same filter at every location</text>
			<path d="M54 170H315M309 164L315 170L309 176" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2"></path>
			<text class="viz-axis-label" x="184" y="188" text-anchor="middle">spatial resolution decreases: 32 &#8594; 16 &#8594; 1</text>
			<path d="M54 204H315M309 198L315 204L309 210" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-dasharray:5 4"></path>
			<text class="viz-axis-label" x="184" y="222" text-anchor="middle">channels and receptive field increase</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow the stated grid and channel dimensions from left to right. These illustrated layers preserve, then downsample, the grid while changing channel width. Global average pooling produces one value per channel. The spatial choices determine which locations remain distinguishable. <span class="visual-reader-check" data-reader-check="cnn-spatial-channel-pyramid"><strong>Check:</strong> Why does spatial pooling produce 1x1x64 rather than one scalar for the entire image?</span></figcaption>
</figure>

## Standard CNN ingredients

- **Spatial convolution:** shared local feature computation.
- **ReLU/GELU:** nonlinear composition rather than one collapsed affine map.
- **Normalization:** an implementation choice, not a prerequisite for every trainable CNN.
- **Pooling/strided convolution:** different downsampling operators with distinct aliasing and information effects.
- **Global average pooling:** one spatial mean per channel, not universal input-transformation invariance.
- **$1\times1$ convolution:** channel mixing at each location.

## Receptive field

Track receptive-field extent $r$ and input-coordinate spacing $j$, starting from $r=j=1$.
One layer gives $r'=r+(k-1)dj$ and $j'=js$ per axis.
Two stride-one $3\times3$ layers therefore span five input positions, not an exponential extent.
Boundary handling and strided sampling qualify translation equivariance; arbitrary one-pixel shifts need not produce one-cell output shifts.

## Change the Requirement

### Preserve Fine Spatial Evidence

**New task:** Retain small-object or boundary information that downsampling removes.

**Picture and change:** Consider higher-resolution paths, suitable dilation, or multiscale features rather than merely widening channels.

**Check:** Two $2\times2$ patches with one active pixel in different locations have the same average, $1/4$.
More channels after that scalar average cannot reconstruct which location was active.

**Use when:** Localization matters; measure compute, aliasing, and dense-task performance together.

### Reduce Channel Work

**New task:** Replace full convolution with depthwise spatial filtering followed by pointwise mixing.

**Picture and change:** For the fixture, weight counts become $3(3)(3)+3(16)=75$, ignoring biases.

**Check:** Arithmetic becomes $32(32)(75)=76800$ MACs, but the restricted factorization cannot represent every full convolution.

**Use when:** Measured hardware cost binds; fewer MACs do not guarantee lower latency.

<!-- learning:core-end -->
<div data-learning-extension>

## Architectural eras

| Era | Architecture | Key idea |
|-----|-------------|---------|
| 2012 | AlexNet | First major win; ReLU + dropout + GPU |
| 2014 | VGG | All $3 \times 3$ convs, very deep |
| 2014 | GoogLeNet / Inception | Multi-scale modules, dim reduction |
| 2015 | ResNet | Residual connections enable 50+ layers |
| 2016 | DenseNet | Dense feature reuse |
| 2017 | MobileNet | Depthwise separable convs for efficiency |
| 2019 | EfficientNet | Compound scaling depth × width × resolution |
| 2020 | ViT | Transformers replace CNN backbones |
| 2022 | ConvNeXt | Modernized ResNet matching ViT performance |

The table names historical design examples, not a measured ranking or deployment census.

## ConvNeXt and the modern CNN

ConvNeXt [(Liu et al., 2022)](https://arxiv.org/abs/2201.03545) modernized ResNet-50 by adopting transformer-era design choices:

- LayerNorm instead of BatchNorm.
- GELU instead of ReLU.
- Larger kernels ($7 \times 7$ depthwise).
- Inverted bottleneck (channels-up then down).

Compare complete training recipes, workloads, quality targets, and hardware before attributing a result solely to convolution versus attention.

## Common pitfalls

- **Shape assumptions.** A $1\times1$ valid convolution need not shrink the grid; stride and padding must be declared.
- **Axis confusion.** Distinguish the chosen tensor layout from the library's default and physical memory layout.
- **Invariance claims.** Pooling averages feature positions; it does not make every input transformation irrelevant.
- **Theoretical versus effective context.** A receptive-field bound lists possible dependencies, not equal influence from every input pixel.

</div>

## Related

- [Residual connections](/concepts/residual-connections/). What made deep CNNs trainable.
- [BatchNorm vs LayerNorm](/concepts/batchnorm-vs-layernorm/). Normalization for vision.
- [Vision transformers](/concepts/vision-transformers/). Alternative paradigm.
