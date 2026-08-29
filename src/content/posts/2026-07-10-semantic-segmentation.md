---
title: "Semantic segmentation"
description: "Assign a class to every pixel: encoder-decoder architectures, losses, IoU, class imbalance, boundaries, and deployment constraints."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Semantic segmentation is the dense version of classification: a class label for every pixel, which is what scene parsing, medical imaging, and autonomous perception actually need. Two things make it hard. A network downsamples to build semantic context, so the central challenge is recovering spatial precision at the boundaries it blurred. And the obvious metric lies: pixel accuracy can look excellent while the model quietly predicts background everywhere. It differs from object detection, which predicts boxes, and instance segmentation, which separates individual objects of the same class.

## Architectures

- **FCN:** replaces dense heads with convolutional prediction.
- **U-Net:** encoder-decoder with skip connections that restore spatial detail.
- **DeepLab:** dilated convolution for multi-scale context without losing resolution.
- **Transformer decoders:** combine global context with learned masks or pixel queries.

The recurring trade-off is semantic context versus precise boundaries.

## Losses

Pixelwise cross-entropy is the baseline. Class-weighted or focal losses handle imbalance. Dice loss rewards overlap directly and is common when the positive region is small. Boundary losses emphasize shape but are sensitive to annotation noise.

## Metrics

Intersection over Union for class $c$ is

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}.$$

Mean IoU averages across classes. Report per-class IoU and boundary quality when rare or safety-critical classes matter, because pixel accuracy can look excellent by predicting background.

## In an interview

1. Separate semantic from instance segmentation.
2. Choose an encoder-decoder and explain how resolution is recovered.
3. Discuss imbalance, annotation quality, and augmentation.
4. Use mIoU plus critical-class and boundary metrics.
5. Cover tiling, latency, memory, and confidence handling at deployment.

## Common confusions

- **"Accuracy is enough."** Background dominance hides the failures that matter.
- **"Upsampling recovers lost detail."** Skip connections or high-resolution features carry the information; interpolation alone cannot.
- **"More precise masks are always better."** Annotation boundaries are often uncertain, and task value may depend on object-level outcomes.

*Related: [CNN architecture](/concepts/cnn-architecture/), [ResNet](/concepts/resnet/), and [object detection](/concepts/object-detection-overview/).*
