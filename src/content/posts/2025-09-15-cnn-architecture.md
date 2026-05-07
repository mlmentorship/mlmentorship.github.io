---
title: "CNN architecture"
description: "Convolutions encode translation equivariance and locality. The structural inductive bias that powered the deep learning revolution in vision."
date: "2025-09-15"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A **convolutional neural network** stacks **convolutional layers** (sliding-window linear operators with shared weights), **non-linearities**, and **pooling / downsampling** to map images to feature maps that grow in semantic abstraction and shrink in spatial resolution with depth.

## Why it matters

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

- [Residual connections](/reference/residual-connections/). What made deep CNNs trainable.
- [BatchNorm vs LayerNorm](/reference/batchnorm-vs-layernorm/). Normalization for vision.
- [Vision transformers](/reference/vision-transformers/). Alternative paradigm.
