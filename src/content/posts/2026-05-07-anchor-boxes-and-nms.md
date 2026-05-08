---
title: "Anchor boxes and non-maximum suppression"
description: "Object detectors predict thousands of overlapping boxes. Anchors give each prediction a prior shape; NMS prunes near-duplicates. The pre-DETR pipeline that defined the field for a decade."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Anchor boxes** are predefined reference boxes at every spatial position; the detector predicts offsets relative to each anchor instead of absolute box coordinates. **Non-maximum suppression** (NMS) post-processes the resulting predictions, keeping only the highest-scoring box from each cluster of overlapping boxes.

## Why it matters

Without anchors, a detector would have to predict box coordinates from scratch at every position, with no prior on shape or scale. With anchors, the network only has to predict small offsets, which is a much easier regression problem. Faster R-CNN, SSD, YOLOv2 to v5, and RetinaNet all use anchor boxes.

Without NMS, the same object would generate dozens of overlapping detections; the metric (mAP) would crater. NMS is the canonical decoder for any detector that produces redundant predictions.

DETR ([Carion et al., 2020](https://arxiv.org/abs/2005.12872)) and its successors removed both, but they remain the dominant design in production detectors today.

## Anchors

At each spatial location $(i, j)$ on the feature map, place $K$ predefined boxes (typically $K = 3$ to $9$) at different scales and aspect ratios. The network output at that location is, per anchor:

- **4 box-regression values**: $\Delta x, \Delta y, \Delta w, \Delta h$ relative to the anchor.
- **$C$ class scores**: probabilities for each class.
- **1 objectness score** (in some designs): is there an object here at all.

Total output channels per location: $K \cdot (4 + C + 1)$.

### Encoding the offsets

The decoder transforms anchor box $(x_a, y_a, w_a, h_a)$ and predicted deltas $(\Delta x, \Delta y, \Delta w, \Delta h)$ into a final box:

$$
x = x_a + w_a \Delta x, \quad y = y_a + h_a \Delta y, \quad w = w_a e^{\Delta w}, \quad h = h_a e^{\Delta h}.
$$

The exponential on width/height keeps them positive without a hard constraint.

### Anchor matching at training time

Each anchor is labeled by IoU with ground-truth boxes:

- IoU > 0.7: positive (regress offsets to the matched box, predict its class).
- IoU < 0.3: negative (predict background).
- In between: ignored.

The class imbalance is severe (10x to 100x more negatives than positives). Two main fixes:

- **Hard negative mining**: pick the worst-classified negatives.
- **Focal loss** ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)): downweight easy negatives so the loss focuses on hard examples. The default in RetinaNet.

## NMS

After the detector outputs $N$ candidate boxes with class scores, run NMS per class:

1. Sort boxes by score.
2. Take the highest-scoring box; remove all other boxes with IoU > $\tau$ against it.
3. Repeat with the next highest-scoring remaining box.
4. Stop when no boxes remain.

Threshold $\tau$ typically 0.45 to 0.5. Lower $\tau$ keeps fewer boxes (more aggressive); higher $\tau$ keeps more.

### Variants

- **Soft NMS** ([Bodla et al., 2017](https://arxiv.org/abs/1704.04503)). Instead of dropping suppressed boxes, decay their scores with a Gaussian or linear function of IoU. Helps when objects genuinely overlap.
- **Class-aware NMS**. Run NMS independently per class. Standard.
- **WBF (Weighted Box Fusion)**. Average overlapping boxes (weighted by score) instead of suppressing. Used in detection ensembles.

## What replaced anchors and NMS

DETR-style detectors output a fixed-size set of predictions (typically 100) and use bipartite matching against ground truth at training time. No anchors, no NMS. Cleaner pipeline, but slower convergence and harder to optimize at small scales. Hybrid designs (DAB-DETR, DINO) reintroduce anchor-like priors for stability.

## Common pitfalls

- **Choosing anchor sizes from defaults instead of the data**. Run k-means on training-set box dimensions to pick anchors.
- **Forgetting per-class NMS**. Two different objects (a dog and a frisbee) can legitimately overlap; class-aware NMS keeps both.
- **Using the same NMS threshold for inference as for evaluation.** mAP definition uses IoU thresholds (e.g. 0.5 to 0.95); NMS is a separate hyperparameter that affects the boxes you generate. Tune them independently.

## Related

- [Object detection: Faster R-CNN, YOLO, DETR](/concepts/object-detection-faster-r-cnn-yolo-detr/).
- [Convolution as matrix multiplication](/concepts/convolution-as-matmul/).
