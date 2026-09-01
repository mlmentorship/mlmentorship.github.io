---
title: "Object detection: Faster R-CNN, YOLO, DETR"
description: "Localize and classify objects in an image. The three main architectural families: two-stage proposal-based, one-stage grid-based, and transformer-based."
date: "2026-03-29"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Object detection** outputs, for each input image, a set of (bounding box, class label, confidence) tuples for the objects present. The three architectural families: **two-stage** (Faster R-CNN family. First propose regions, then classify), **one-stage** (YOLO, RetinaNet. Predict box and class in one pass at every grid cell), and **set-prediction** (DETR. Transformer encoder-decoder predicts a fixed-size set of boxes).

Detection is the **core production CV task**: autonomous driving, retail analytics, surveillance, medical imaging, robotics. The architectural choice determines latency-accuracy tradeoffs and what's possible in your inference budget.

## Two-stage: Faster R-CNN [(Ren et al., 2015)](https://arxiv.org/abs/1506.01497)

Architecture:

1. **Backbone** (ResNet, ConvNeXt) extracts feature map.
2. **Region Proposal Network (RPN)** slides a small network over the feature map, predicts (objectness, box-refinement) at each anchor (preset boxes of various scales / aspect ratios).
3. **RoI Pooling / RoIAlign**: extract fixed-size feature for each top-K proposed region.
4. **Classification + bbox regression head**: per region.

**Strengths**: high accuracy; standard for COCO leaderboards; basis for Mask R-CNN (adds segmentation head).

**Weaknesses**: slow (two stages, hundreds of region computations per image); 5–30 FPS on a GPU.

**Used when**: accuracy matters more than latency; offline analytics; medical imaging.

## One-stage: YOLO ([Redmon et al., 2015](https://arxiv.org/abs/1506.02640) onward), RetinaNet [(Lin 2017)](https://arxiv.org/abs/1708.02002)

Predict boxes and classes directly at every grid cell of the feature map, in a single forward pass:

- **YOLO v1**: $S \times S$ grid, each cell predicts B bounding boxes + class.
- **YOLO v3**: multi-scale predictions across three feature-map levels.
- **YOLO v5/v7/v8/v10/v12**: ongoing engineering improvements (anchor-free, attention, distillation).
- **RetinaNet**: Focal Loss to handle the extreme class imbalance between foreground and background anchors.

**Strengths**: real-time (60–300+ FPS); simpler to deploy.

**Weaknesses**: historically lower accuracy than two-stage; gap closed by ~2020.

**Used when**: real-time required; embedded / edge deployment; autonomous driving.

## Set prediction: DETR [(Carion et al., 2020)](https://arxiv.org/abs/2005.12872)

A transformer encoder-decoder that outputs a **fixed set** of $N$ predictions (e.g., $N = 100$):

1. CNN backbone extracts features.
2. Flatten to a sequence; transformer encoder processes.
3. Transformer decoder takes $N$ learned object queries; cross-attends to encoder features.
4. Each query produces (box, class). Including a "no object" class for unused queries.
5. Loss: bipartite matching (Hungarian) between predictions and ground-truth boxes.

**Strengths**: no anchors, no NMS; cleaner formulation; scales with transformer pretraining.

**Weaknesses**: slow training convergence; lower throughput than YOLO. **Deformable DETR**, **DINO** (DETR with denoising), **Co-DETR** addressed convergence and accuracy.

**Used when**: research / leaderboard work; integration with vision-language pretraining (Grounding DINO, OWL-ViT).

<!-- visual:detector-candidate-resolution -->
<figure class="learning-figure" aria-labelledby="detector-resolution-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="detector-resolution-title">Where does each detector family resolve duplicate predictions?</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison of inference-time non-maximum suppression and DETR's training-time one-to-one assignment">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 264" role="img" aria-labelledby="detector-nms-title detector-nms-desc">
				<title id="detector-nms-title">Classic proposal and dense detectors resolve duplicates after prediction</title>
				<desc id="detector-nms-desc">Faster R-CNN proposals and the dense predictions used by RetinaNet and most YOLO versions can produce three overlapping boxes for one object. At inference, non-maximum suppression keeps the highest-scoring box and removes the two lower-scoring duplicates.</desc>
				<text class="viz-axis-label" x="150" y="17" text-anchor="middle">CLASSIC PROPOSAL / DENSE PIPELINE</text>
				<rect class="viz-plot-bg" x="9" y="28" width="282" height="226" rx="5"></rect>
				<text class="viz-label" x="150" y="48" text-anchor="middle">many scored candidates for one object</text>
				<rect x="83" y="61" width="116" height="66" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:4"></rect>
				<rect x="94" y="67" width="116" height="66" rx="3" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:8 5"></rect>
				<rect x="73" y="69" width="116" height="66" rx="3" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-dasharray:2 5"></rect>
				<text class="viz-callout" x="77" y="59">0.94 · highest</text>
				<text class="viz-axis-label" x="150" y="148" text-anchor="middle">overlap is expected, not three objects</text>
				<path d="M150 155V169M144 163L150 169L156 163" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<rect x="77" y="174" width="146" height="31" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect>
				<text class="viz-callout" x="150" y="194" text-anchor="middle">INFERENCE · NMS keeps 0.94</text>
				<path d="M150 205V219M144 213L150 219L156 213" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<rect x="112" y="224" width="76" height="22" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:3"></rect>
				<text class="viz-axis-label" x="150" y="240" text-anchor="middle">one kept box</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 264" role="img" aria-labelledby="detector-detr-title detector-detr-desc">
				<title id="detector-detr-title">DETR learns distinct prediction slots before direct inference</title>
				<desc id="detector-detr-desc">Four learned object-query slots predict a dog box, a car box, and two no-object values in parallel. During training only, Hungarian bipartite matching assigns at most one query to each ground-truth object. At inference, the slots directly emit the final set and there is no NMS gate.</desc>
				<text class="viz-axis-label" x="150" y="17" text-anchor="middle">DETR · SET-PREDICTION PIPELINE</text>
				<rect class="viz-plot-bg" x="9" y="28" width="282" height="226" rx="5"></rect>
				<text class="viz-label" x="150" y="48" text-anchor="middle">fixed learned query slots predict in parallel</text>
				<rect x="30" y="59" width="48" height="27" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
				<rect x="94" y="59" width="48" height="27" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
				<rect x="158" y="59" width="48" height="27" rx="4" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-dasharray:5 3"></rect>
				<rect x="222" y="59" width="48" height="27" rx="4" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-dasharray:5 3"></rect>
				<text class="viz-callout" x="54" y="77" text-anchor="middle">Q1</text>
				<text class="viz-callout" x="118" y="77" text-anchor="middle">Q2</text>
				<text class="viz-callout" x="182" y="77" text-anchor="middle">Q3</text>
				<text class="viz-callout" x="246" y="77" text-anchor="middle">Q4</text>
				<path d="M54 86V103M118 86V103M182 86V103M246 86V103" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<rect x="30" y="105" width="48" height="30" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:3"></rect>
				<rect x="94" y="105" width="48" height="30" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:3"></rect>
				<rect x="158" y="105" width="48" height="30" rx="3" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-dasharray:5 3"></rect>
				<rect x="222" y="105" width="48" height="30" rx="3" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-dasharray:5 3"></rect>
				<text class="viz-axis-label" x="54" y="124" text-anchor="middle">dog box</text>
				<text class="viz-axis-label" x="118" y="124" text-anchor="middle">car box</text>
				<text class="viz-axis-label" x="182" y="119" text-anchor="middle">no</text>
				<text class="viz-axis-label" x="182" y="130" text-anchor="middle">object</text>
				<text class="viz-axis-label" x="246" y="119" text-anchor="middle">no</text>
				<text class="viz-axis-label" x="246" y="130" text-anchor="middle">object</text>
				<path d="M54 146V163M118 146V163M54 155H118" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-dasharray:5 3"></path>
				<text class="viz-axis-label" x="150" y="177" text-anchor="middle">TRAINING ONLY · one-to-one Hungarian match</text>
				<path d="M150 184V201M144 195L150 201L156 195" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<rect x="66" y="206" width="168" height="40" rx="4" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect>
				<text class="viz-callout" x="150" y="223" text-anchor="middle">INFERENCE · direct final set</text>
				<text class="viz-axis-label" x="150" y="239" text-anchor="middle">dog + car · no NMS gate</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> classic proposal and dense detectors may score several boxes for the same object, so NMS removes lower-scoring overlaps at inference. DETR moves uniqueness into training: Hungarian matching gives each ground-truth object at most one query slot, unused slots learn “no object,” and inference emits the resulting set directly.</figcaption>
</figure>

## Key shared components

### Bounding box parameterization
Either $(x_\text{center}, y_\text{center}, w, h)$ or $(x_\text{min}, y_\text{min}, x_\text{max}, y_\text{max})$. Critical for loss design.

### IoU loss
Intersection over Union directly measures box overlap. Variants: GIoU, DIoU, CIoU. Handle non-overlapping boxes better than naive L1 / L2 on coordinates.

### Non-Maximum Suppression (NMS)
Post-processing for anchor-based detectors: remove duplicate boxes for the same object by keeping highest-confidence and suppressing others with IoU > threshold (typically 0.5). DETR-family removes the need for NMS by design.

### Anchors
Predefined boxes of various scales / aspect ratios. The detector predicts offsets from these. **Anchor-free** methods (FCOS, CenterNet, YOLOv8) predict box centers directly.

## Datasets and metrics

- **PASCAL VOC** (legacy): 20 classes; mAP@IoU=0.5.
- **COCO**: 80 classes; metric **mAP** averaged over IoU thresholds 0.5 → 0.95 (the dominant 2026 benchmark).
- **Open Images, LVIS**: large-vocabulary detection.

The standard evaluation: average precision per class, then mean across classes (mAP).

## Production considerations

- **Latency**: YOLO-class models for real-time; Faster R-CNN for accuracy.
- **Class imbalance**: many backgrounds, few foregrounds. Focal loss (RetinaNet) addresses this.
- **Small objects**: hardest case; multi-scale features (FPN. Feature Pyramid Network) help.
- **Open vocabulary**: align detector classes with text embeddings (CLIP-style) for zero-shot detection (OWL-ViT, Grounding DINO).

## Common pitfalls

- **Training on small datasets without strong augmentation.** Detection is data-hungry; mosaic, mixup, copy-paste augmentations standard.
- **Confusing IoU thresholds.** mAP@0.5 ≠ mAP@0.5:0.95; specify which.
- **Forgetting NMS / NMS thresholds.** Setting NMS too aggressive merges distinct objects; too loose duplicates them.
- **Using the wrong evaluation tool.** COCO eval and VOC eval differ; use the one for your reporting benchmark.
- **Treating YOLO as a single algorithm.** Many YOLO versions exist with very different performance; cite the specific version.

## Related

- [CNN architecture](/concepts/cnn-architecture/). Backbone.
- [Vision transformers](/concepts/vision-transformers/). Alternative backbone.
