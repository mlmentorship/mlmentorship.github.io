---
title: "CNN architecture"
description: "Convolutions encode translation equivariance and locality. The structural inductive bias that powered the deep learning revolution in vision."
date: "2025-09-15"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **convolutional neural network** stacks **convolutional layers** (sliding-window linear operators with shared weights), **non-linearities**, and **pooling / downsampling** to map images to feature maps that grow in semantic abstraction and shrink in spatial resolution with depth.

CNNs powered the deep-learning revolution in computer vision (AlexNet 2012, VGG 2014, ResNet 2015). Their structural priors. Translation equivariance via weight sharing, local receptive fields, hierarchical composition. Match the structure of natural images and gave them a huge sample-efficiency advantage over fully-connected networks. Even in the transformer era, modern CNNs (ConvNeXt) remain competitive on standard benchmarks.

## The building block: convolutional layer

Apply a small filter (e.g., $3 \times 3 \times C_{\text{in}}$) at every spatial position of the input, producing one output channel. Repeat with $C_{\text{out}}$ filters → output of shape $H \times W \times C_{\text{out}}$.

Per output:

$$
y_{ij,c_{\text{out}}} = \sum_{c_{\text{in}}} \sum_{u, v} W_{u, v, c_{\text{in}}, c_{\text{out}}} \cdot x_{i+u, j+v, c_{\text{in}}} + b_{c_{\text{out}}}.
$$

Critical properties:

- **Weight sharing**: the same filter is applied at every position. Vastly fewer parameters than fully-connected.
- **Translation equivariance**: shifting the input shifts the output by the same amount. Hard-coded inductive bias.
- **Locality**: each output depends only on a small spatial neighborhood of the input.

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
	<figcaption><strong>Read it this way:</strong> Move left to right. Convolution preserves the image grid while detecting the same local pattern everywhere. Later stages usually downsample height and width, add feature channels, and combine earlier neighborhoods, so each unit summarizes more of the original image. Global average pooling collapses the remaining locations before classification. Dimensions are illustrative.</figcaption>
</figure>

## Standard CNN ingredients

- **Conv $3 \times 3$**: workhorse; captures local features.
- **ReLU / GELU**: pointwise non-linearity.
- **Batch normalization**: stabilizes training, allows higher LRs.
- **Max pooling / average pooling**: downsample by taking max / average over $2 \times 2$ windows.
- **Strided convolution**: alternative downsampling that learns the filter.
- **Global average pooling**: reduce $H \times W \times C$ to $1 \times 1 \times C$ before the classifier head.
- **$1 \times 1$ convolution**: per-pixel linear projection across channels; cheap channel mixing.

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

In 2026, **ViT and ConvNeXt are the dominant ImageNet-class backbones**; classic ResNet-50 still ubiquitous in transfer-learning pipelines.

## Receptive field

The receptive field of a unit is the spatial extent of the input that influences it. Stacking $L$ layers of $3 \times 3$ conv with stride 1 gives receptive field $1 + 2L$. Pooling and strided convolution multiply the effective stride, growing the receptive field exponentially.

For dense prediction (segmentation), large receptive field matters; for classification, global pooling at the end aggregates over all spatial locations.

## ConvNeXt and the modern CNN

ConvNeXt [(Liu et al., 2022)](https://arxiv.org/abs/2201.03545) modernized ResNet-50 by adopting transformer-era design choices:

- LayerNorm instead of BatchNorm.
- GELU instead of ReLU.
- Larger kernels ($7 \times 7$ depthwise).
- Inverted bottleneck (channels-up then down).

Result: matches or beats ViT on ImageNet at the same compute. CNNs are not obsolete; transformers won by being *better-designed*, not by inherent architectural superiority.

## Common pitfalls

- **Forgetting padding.** Without padding, each conv layer shrinks $H, W$. Use `padding='same'` to preserve spatial dimensions.
- **Channels-first vs. channels-last.** PyTorch defaults to channels-first $(B, C, H, W)$; TensorFlow / Keras to channels-last. Conversions are common bug sources.
- **Skipping batch normalization.** Deep CNNs without BN are very hard to train.
- **Using max-pool too aggressively.** Halving spatial resolution at every layer destroys fine detail; stride-2 conv blocks let you control it.
- **Treating CNNs as universally outperformed by transformers.** They are competitive on ImageNet at scale; on small datasets, CNNs often beat ViT due to inductive biases.

## Related

- [Residual connections](/concepts/residual-connections/). What made deep CNNs trainable.
- [BatchNorm vs LayerNorm](/concepts/batchnorm-vs-layernorm/). Normalization for vision.
- [Vision transformers](/concepts/vision-transformers/). Alternative paradigm.
