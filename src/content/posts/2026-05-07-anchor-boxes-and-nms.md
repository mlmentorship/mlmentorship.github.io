---
title: "Anchor boxes and non-maximum suppression"
description: "Decode reference-box offsets, trace exact score-ordered suppression and separate training, inference and evaluation overlap rules."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Decode boxes from a declared reference, then suppress overlaps against selected score leaders. NMS removes predictions; it does not cluster object identities.

## Problem

An anchor has center $(10,20)$, size $(4,8)$ and predicted deltas $(0.5,-0.25,\log 2,0)$.
Decode its center and size.

Separately, three same-class detections use corner coordinates $(x_1,y_1,x_2,y_2)$:

| Box | Coordinates | Score |
| --- | --- | ---: |
| A | [0,0,10,10] | 0.94 |
| B | [2,0,12,10] | 0.82 |
| C | [20,0,30,10] | 0.76 |

Use continuous-coordinate area and strict suppression at $\operatorname{IoU}>0.50$.
Which original indices survive?

### Encoding the offsets

The decoder transforms anchor box $(x_a, y_a, w_a, h_a)$ and predicted deltas $(\Delta x, \Delta y, \Delta w, \Delta h)$ into a final box:

$$
x = x_a + w_a \Delta x, \quad y = y_a + h_a \Delta y, \quad w = w_a e^{\Delta w}, \quad h = h_a e^{\Delta h}.
$$

The decoded center is $(12,18)$ and size $(8,8)$.
Exponentiation gives positive sizes mathematically; implementations still need scale/clamp, overflow and invalid-box policies.

## NMS

After score filtering, process the declared class groups:

1. Sort descending by score; break ties by original input index in this example.
2. Keep the leader; suppress remaining boxes whose IoU with it exceeds $\tau$.
3. Repeat using only remaining candidates.
4. Stop at exhaustion or the declared output limit.

A/B intersection is $8\times10=80$ and union is $100+100-80=120$.
Thus $\operatorname{IoU}=2/3>0.50$: keep A, suppress B, then keep disjoint C.

<!-- visual:nms-greedy-suppression -->
<figure class="learning-figure" aria-labelledby="nms-greedy-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="nms-greedy-title">Which detections survive score-ordered NMS at τ = 0.50?</p>
	<div class="visual-grid--two" role="group" aria-label="Before-and-after comparison of greedy non-maximum suppression">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 210" role="img" aria-labelledby="nms-candidates-title nms-candidates-desc">
				<title id="nms-candidates-title">Three scored candidate boxes before NMS</title>
				<desc id="nms-candidates-desc">A scores 0.94, B 0.82 and C 0.76. Drawn A and B each have area 6400, with intersection 5120 and union 7680, so IoU is exactly two-thirds. C is disjoint. Drawing coordinates are x equals 20 plus eight times image x, and y equals 65 plus eight times image y. Suppression uses strict IoU greater than 0.50.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="175" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">CANDIDATES · sorted A → B → C</text>
				<rect x="36" y="65" width="64" height="80" rx="0" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1"></rect>
				<rect x="20" y="65" width="80" height="80" rx="0" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:4"></rect>
				<rect x="36" y="65" width="80" height="80" rx="0" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:8 5"></rect>
				<rect x="180" y="65" width="80" height="80" rx="0" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:4"></rect>
				<text class="viz-callout" x="20" y="56">A · 0.94</text>
				<text class="viz-callout" x="36" y="166">B · 0.82 · dashed</text>
				<text class="viz-callout" x="180" y="56">C · 0.76</text>
				<text class="viz-label" x="68" y="102" text-anchor="middle">A ∩ B</text>
				<text class="viz-axis-label" x="68" y="121" text-anchor="middle">IoU = 2/3</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 210" role="img" aria-labelledby="nms-result-title nms-result-desc">
				<title id="nms-result-title">Boxes retained after greedy NMS</title>
				<desc id="nms-result-desc">A is selected first. B is crossed out because its IoU with A is two-thirds, strictly above 0.50. Disjoint C remains and is selected next. Original coordinates and indices A and C are retained, not averaged.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="175" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">RESULT · keep A, suppress B, keep C</text>
				<rect x="20" y="65" width="80" height="80" rx="0" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:5"></rect>
				<rect x="36" y="65" width="80" height="80" rx="0" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:8 5;opacity:.38"></rect>
				<path d="M42 71L110 139M110 71L42 139" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:4"></path>
				<rect x="180" y="65" width="80" height="80" rx="0" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:5"></rect>
				<text class="viz-callout" x="20" y="56">A · KEEP FIRST</text>
				<text class="viz-callout" x="36" y="166">B · SUPPRESS</text>
				<text class="viz-callout" x="180" y="56">C · KEEP NEXT</text>
				<text class="viz-axis-label" x="150" y="194" text-anchor="middle">retained indices, descending score: [A, C]</text>
			</svg>
		</section>
	</div>
	<div class="plot-panel">
		<svg viewBox="0 0 300 100" role="img" aria-labelledby="nms-chain-title nms-chain-desc">
			<title id="nms-chain-title">A connected overlap chain can retain both endpoints</title>
			<desc id="nms-chain-desc">For this separate chain check only, move C to image coordinates 4,0,14,10. A-B and B-C each have IoU two-thirds; A-C has IoU three-sevenths. Lines denote overlap above 0.50. Selecting A suppresses B, but C survives even though the three nodes share a connected component.</desc>
			<path d="M90 38H120M180 38H210" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<rect class="viz-node viz-node--output" x="30" y="18" width="60" height="40" rx="3"></rect>
			<rect class="viz-node viz-node--warning" x="120" y="18" width="60" height="40" rx="3"></rect>
			<rect class="viz-node viz-node--output" x="210" y="18" width="60" height="40" rx="3"></rect>
			<text class="viz-callout" x="60" y="35" text-anchor="middle">A</text><text class="viz-label" x="60" y="51" text-anchor="middle">keep</text>
			<text class="viz-callout" x="150" y="35" text-anchor="middle">B</text><text class="viz-label" x="150" y="51" text-anchor="middle">drop</text>
			<text class="viz-callout" x="240" y="35" text-anchor="middle">C</text><text class="viz-label" x="240" y="51" text-anchor="middle">keep</text>
			<text class="viz-label" x="150" y="83" text-anchor="middle">chain check: C moves to [4,0,14,10]</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> keep A, suppress B at IoU 2/3, then keep C. In the separate chain, A and C also survive; NMS is not one survivor per connected component.</figcaption>
</figure>

## Change 1: Different Objects Overlap

**New task.** Crowded scenes lose true objects.

**Picture and change.** Compare class-aware suppression, Soft-NMS or another justified duplicate policy using recall and false positives.

**Check.** Overlap is not proof of one object.

**Use when.** Crowding causes suppression errors.

## Change 2: Default Anchor Sizes Miss the Data

**New task.** Object scales differ from reference shapes.

**Picture and change.** Choose training-only shapes/matching or an anchor-free head; recheck decode and small-object coverage.

**Check.** Anchor-free need not mean NMS-free.

**Use when.** Reference geometry mismatches the task.

## Failure: Three Thresholds Are Treated as One

Training matches label examples; inference NMS filters predictions; evaluation matches predictions to truth.
Their IoU thresholds have different meanings and require separate decisions.

## Transfer Check

Does NMS average retained and suppressed boxes?

### Reasoned Answer

No. Greedy NMS retains original selected boxes. Weighted fusion is another operation.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Anchors

$K$ references can emit four offsets, $C$ class outputs and, in some heads, objectness.
Then $K(4+C+1)$ is a specific output convention, not universal.
Specify logits/probabilities, score combination, coordinate transforms and area policy.

### Anchor matching at training time

Illustrative matches: IoU above 0.7 positive, below 0.3 background, intermediate ignored.
Specify boundaries, forced matches and ties for the actual detector.
Hard-negative mining selects difficult negatives; focal loss reduces easy-example weight under its definition.

### Variants

| Method | Change |
| --- | --- |
| Soft-NMS | Decay/reorder scores rather than hard removal; evaluate crowded scenes |
| Class-aware | Separate classes; shared-object/category policy still matters |
| Weighted fusion | Combine coordinates under explicit ensemble/grouping assumptions |

Naive pairwise NMS costs $O(N^2)$ worst-case; sorting/top-K costs are separate.
Lower thresholds suppress more against a fixed leader, not necessarily fewer total greedy survivors.
For a separate counterexample, use descending scores A, B, C, D:

| Box | Coordinates |
| --- | --- |
| A | [0,0,10,10] |
| B | [4,0,14,10] |
| C | [4,-3,14,7] |
| D | [4,3,14,13] |

| Threshold | Survivors |
| --- | --- |
| 0.50 | A, B |
| 0.30 | A, C, D |

## What replaced anchors and NMS

Set-prediction designs such as DETR use training-time one-to-one matching and a specified slot budget.
Learned references and duplicate policies differ across versions; do not infer production prevalence or universal quality.

## Common pitfalls

Define empty/invalid boxes, ties, strict inequalities, class mode, filtering and output limits before comparing implementations.

## Sources

- [DETR, 2020](https://arxiv.org/abs/2005.12872).
- [Focal loss, 2017](https://arxiv.org/abs/1708.02002).
- [Soft-NMS, 2017](https://arxiv.org/abs/1704.04503).

## Related

- [Object detection: Faster R-CNN, YOLO, DETR](/concepts/object-detection-overview/).
- [Convolution as matrix multiplication](/concepts/convolution-as-matmul/).

</div>
