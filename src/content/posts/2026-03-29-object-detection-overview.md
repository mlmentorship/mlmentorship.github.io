---
title: "Object detection: Faster R-CNN, YOLO, DETR"
description: "Localize and classify objects in an image. The three main architectural families: two-stage proposal-based, one-stage grid-based, and transformer-based."
date: "2026-03-29"
draft: false
tags: ["concepts"]
category: "concepts"
---

Object detection constructs a scored set of boxes; candidate generation, training assignment and duplicate handling are separate choices.

## Summary

Where are candidates generated, assigned to target objects, and converted into the final prediction set?
Declare box coordinates, classes, score meaning and the maximum output count.
Background or no-object predictions are not additional objects.

## Two-stage: Faster R-CNN [(Ren et al., 2015)](https://arxiv.org/abs/1506.01497)

Extract backbone features, generate RPN objectness/box proposals, then classify and refine selected regions.
RoI pooling quantizes regions; RoIAlign samples aligned features without that coordinate quantization.
Proposal and final duplicate handling belong to the specified implementation.
Mask R-CNN adds a mask head; it does not change this comparison into a speed guarantee.

## One-stage: YOLO ([Redmon et al., 2015](https://arxiv.org/abs/1506.02640) onward), RetinaNet [(Lin 2017)](https://arxiv.org/abs/1708.02002)

Dense heads predict from feature locations without a separate region-classification stage.
YOLO v1's grid differs from v3's multiscale predictions and later anchor-free or assignment variants.
Name the version and head, not merely the family.
RetinaNet uses focal loss to reduce easy-background dominance.
Anchor-free parameterization does not imply NMS-free inference.

## Set prediction: DETR [(Carion et al., 2020)](https://arxiv.org/abs/2005.12872)

Learned object queries attend to encoded image features and predict boxes/classes, including no-object.
During training, one-to-one matching assigns targets to distinct slots; unmatched queries receive the specified no-object loss.
Inference does not match against unavailable ground truth.
Original DETR emits slots without an NMS stage, but training incentives do not mathematically forbid all duplicate predictions.

<!-- visual:detector-candidate-resolution -->
<figure class="learning-figure" aria-labelledby="detector-resolution-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="detector-resolution-title">Training-time matching and inference-time duplicate removal are different operations</p><div class="visual-argument" data-visual-argument="detector-candidate-resolution" data-argument-kind="matrix"><p class="visual-argument-premise">Training assignment and inference suppression solve different problems</p><svg viewBox="0 0 360 142" role="img" aria-label="Training assignment and inference suppression solve different problems. Training matches predictions to ground-truth supervision. Inference removes redundant predicted boxes without access to ground truth."><title>Training assignment and inference suppression solve different problems</title><desc>Training matches predictions to ground-truth supervision. Inference removes redundant predicted boxes without access to ground truth.</desc><text x="160.64499999999998" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="160.64499999999998" dy="0">truth used</tspan></text><text x="280.215" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="280.215" dy="0">operation</tspan></text><text x="88.86" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="88.86" dy="0">training</tspan></text><rect data-proof-cell="0,0" x="100.86" y="36" width="116.57" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="159.14499999999998" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="159.14499999999998" dy="0">yes</tspan></text><rect data-proof-cell="0,1" x="220.43" y="36" width="116.57" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="278.71500000000003" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="278.71500000000003" dy="0">match</tspan></text><text x="88.86" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="88.86" dy="0">inference</tspan></text><rect data-proof-cell="1,0" x="100.86" y="83" width="116.57" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="159.14499999999998" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="159.14499999999998" dy="0">no</tspan></text><rect data-proof-cell="1,1" x="220.43" y="83" width="116.57" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="278.71500000000003" y="110.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="278.71500000000003" dy="0">suppress</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">the left lane resolves scored overlaps with inference-time NMS. The right lane assigns targets one-to-one during training and predicts a set at inference. That objective discourages duplicates; it does not guarantee perfect uniqueness.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="detector-candidate-resolution"><strong>Check:</strong> Which duplicate-resolution operation is used during inference in each pictured pipeline? Which matching exists only in training?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="detector-candidate-resolution">Training matches predictions to ground-truth supervision. Inference removes redundant predicted boxes without access to ground truth.</span></figcaption>
</figure>

## Work the assignment

| Matching cost | Target one | Target two |
| --- | --- | --- |
| Query one | 1 | 2 |
| Query two | 2 | 100 |

Taking query one's cheapest target first leaves total cost $1+100=101$.
Swapping assignments gives $2+2=4$, the global minimum.
Matching cost selects pairs; subsequent training losses and their weights must be defined separately.
This equal-count toy omits unmatched-query costs.

## Change the deployment requirement

### Meet a request deadline

**New task:** complete an edge request within thirty milliseconds.

**Picture and change:** a hypothetical serial trace takes four ms preprocessing, eighteen ms model time and ten ms postprocessing.

**Check:** total time is 32 ms, so the model-only timing hides a miss.
A compatible twelve-ms model gives 26 ms on that same assumed trace.
Measure accuracy, memory and actual end-to-end tails; component percentiles cannot simply be added.

**Use when:** service constraints change; family labels alone do not establish FPS or quality.

### Prioritize small or crowded objects

**New task:** one-pixel localization errors now determine usefulness.

**Picture and change:** shift a two-by-two box one pixel horizontally: IoU becomes $2/6=1/3$.

**Check:** the same shift of a twenty-by-twenty box gives $380/420=19/21$.
Evaluate small/crowded slices and consider resolution, multiscale features and duplicate policy.
Text-conditioned classes additionally need grounding checks; a vocabulary match is not localization evidence.

**Use when:** the object distribution or class vocabulary differs from the benchmark average.

## Reconstruct the distinction

Does using no anchors remove duplicate-resolution requirements?

<details>
<summary>Reasoned answer</summary>

No. Parameterization, training assignment and inference selection are independent design axes.
An anchor-free head can still use NMS.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Key shared components

### Bounding box parameterization

Declare center/size versus corner coordinates, normalization, resizing and padding transforms before losses or evaluation.

### IoU loss

IoU compares intersection to union; disjoint boxes have zero IoU.
GIoU adds enclosure geometry; DIoU uses center distance; CIoU includes an aspect-ratio term.
These are different objectives, not interchangeable names for ordinary coordinate L1 loss.

### Non-Maximum Suppression (NMS)

Specify score ordering, class grouping, IoU threshold and maximum detections.
Aggressive suppression can remove distinct nearby objects.
See the anchor/NMS owner for exact decode and threshold arithmetic.

### Anchors

Anchors provide reference geometry for offsets; anchor-free heads use other coordinate conventions.
Neither alone specifies assignment or postprocessing.

## Datasets and metrics

VOC, COCO, Open Images and LVIS use different classes and evaluation conventions.
Use the official evaluator and state IoU thresholds, interpolation, area slices, class averaging and maximum detections.
AP integrates a ranked precision-recall curve under that protocol; duplicates do not universally reduce AP in every score order.

## Production considerations

FPN supplies multiscale features; test whether the small-object slice improves.
Deformable DETR, DINO and related variants change computation or training; retain versioned comparisons.
Open-vocabulary models add text-conditioned class support, not guaranteed grounding for arbitrary new objects.

## Common pitfalls

An unspecified FPS number omits hardware, resolution, batching and postprocessing.
Keep annotation policy, augmentation and class/area coverage visible when comparing detectors.

## Related

- [CNN architecture](/concepts/cnn-architecture/). Backbone.
- [Vision transformers](/concepts/vision-transformers/). Alternative backbone.

</div>
