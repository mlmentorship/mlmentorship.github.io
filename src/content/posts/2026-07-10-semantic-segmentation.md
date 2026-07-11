---
title: "Semantic segmentation"
description: "Assign a class to every pixel: encoder–decoder architectures, losses, IoU, class imbalance, boundaries, and deployment constraints."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Definition

Semantic segmentation predicts a class label for each pixel. It differs from object detection, which predicts boxes, and instance segmentation, which separates individual objects of the same class.

## Architectures

- **FCN:** replaces dense heads with convolutional prediction.
- **U-Net:** encoder–decoder with skip connections that restore spatial detail.
- **DeepLab:** dilated convolution and multi-scale context.
- **Transformer decoders:** combine global context with learned masks or pixel queries.

The central trade-off is semantic context versus precise boundaries.

## Losses

Pixelwise cross-entropy is the baseline. Class-weighted or focal losses help imbalance. Dice loss directly rewards overlap and is common when positive regions are small. Boundary losses emphasize shape but can be sensitive to annotation noise.

## Metrics

Intersection over Union for class $c$ is

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}.$$

Mean IoU averages across classes. Report per-class IoU and boundary quality when rare or safety-critical classes matter. Pixel accuracy can look excellent by predicting background.

## Interview answer

1. Clarify semantic versus instance segmentation.
2. Choose an encoder–decoder and explain resolution recovery.
3. Discuss imbalance, annotation quality, and augmentation.
4. Use mIoU plus critical-class and boundary metrics.
5. Cover tiling, latency, memory, and confidence handling at deployment.

## Common confusions

- **“Accuracy is enough.”** Background dominance hides failures.
- **“Upsampling recovers lost detail.”** Skip connections or high-resolution features provide information; interpolation alone cannot.
- **“More precise masks are always better.”** Annotation boundaries may be uncertain and task value may depend on object-level outcomes.

*Related: [CNN architecture](/concepts/cnn-architecture/), [ResNet](/concepts/resnet/), and [object detection](/concepts/object-detection-overview/).*
