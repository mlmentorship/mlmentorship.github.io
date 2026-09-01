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

**Learning objective:** trace why upsampling restores output size but cannot by itself recover an erased object boundary, then identify what a same-scale encoder skip contributes.

<!-- visual:semantic-segmentation-boundary-fusion -->
<figure class="learning-figure" aria-labelledby="segmentation-boundary-heading">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="segmentation-boundary-heading">Why does a decoder need fine features, not only more pixels?</p>
	<svg viewBox="0 0 360 448" role="img" aria-labelledby="segmentation-boundary-title segmentation-boundary-desc">
		<title id="segmentation-boundary-title">Downsampling loses exact boundary location, while an encoder skip supplies fine evidence to the decoder</title>
		<desc id="segmentation-boundary-desc">A fine eight-column input feature grid contains an object edge between columns five and six. Encoding and downsampling produce a coarse four-column semantic grid where the edge occupies an uncertain cell. Upsampling that coarse grid creates a larger mask but leaves a wide dashed uncertain boundary. A dashed skip path carries same-scale fine encoder features around the bottleneck; fusing those features with coarse semantics places a sharp solid boundary between columns five and six.</desc>
		<defs>
			<marker id="segmentation-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="viz-arrow-forward"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="28" width="344" height="412" rx="5"></rect>
		<text class="viz-axis-label" x="16" y="18">SPATIAL SIZE IS NOT THE SAME AS SPATIAL EVIDENCE</text>
		<text class="viz-axis-label" x="22" y="53">1 · FINE ENCODER FEATURES</text>
		<text class="viz-label" x="22" y="70">high resolution · exact edge retained</text>
		<g>
			<rect class="viz-node viz-node--input" x="22" y="81" width="38" height="38"></rect>
			<rect class="viz-node viz-node--input" x="60" y="81" width="38" height="38"></rect>
			<rect class="viz-node viz-node--input" x="98" y="81" width="38" height="38"></rect>
			<rect class="viz-node viz-node--input" x="136" y="81" width="38" height="38"></rect>
			<rect class="viz-node viz-node--input" x="174" y="81" width="38" height="38"></rect>
			<rect class="viz-node" x="212" y="81" width="38" height="38"></rect>
			<rect class="viz-node" x="250" y="81" width="38" height="38"></rect>
			<rect class="viz-node" x="288" y="81" width="38" height="38"></rect>
			<path d="M 212 78 L 212 122" fill="none" stroke="var(--c-text)" stroke-width="3"></path>
			<text class="viz-callout" x="212" y="136" text-anchor="middle">edge at one pixel boundary</text>
		</g>
		<path d="M 174 143 L 174 167" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#segmentation-arrow)"></path>
		<text class="viz-edge-label" x="208" y="158">encode + downsample</text>
		<text class="viz-axis-label" x="22" y="186">2 · COARSE SEMANTICS</text>
		<text class="viz-label" x="22" y="203">object recognized · edge location compressed</text>
		<rect class="viz-node viz-node--output" x="68" y="214" width="56" height="44"></rect>
		<rect class="viz-node viz-node--output" x="124" y="214" width="56" height="44"></rect>
		<rect class="viz-node viz-node--focus" x="180" y="214" width="56" height="44" stroke-dasharray="5 3"></rect>
		<rect class="viz-node" x="236" y="214" width="56" height="44"></rect>
		<text class="viz-node-value" x="96" y="240">object</text>
		<text class="viz-node-value" x="152" y="240">object</text>
		<text class="viz-node-value" x="208" y="233">edge?</text>
		<text class="viz-node-value" x="208" y="247">wide cell</text>
		<text class="viz-node-value" x="264" y="240">background</text>
		<path d="M 174 266 L 174 290" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#segmentation-arrow)"></path>
		<text class="viz-edge-label" x="211" y="281">upsample alone</text>
		<text class="viz-axis-label" x="22" y="309">3 · LARGER GRID, SAME UNCERTAINTY</text>
		<rect class="viz-node viz-node--output" x="22" y="320" width="38" height="38"></rect>
		<rect class="viz-node viz-node--output" x="60" y="320" width="38" height="38"></rect>
		<rect class="viz-node viz-node--output" x="98" y="320" width="38" height="38"></rect>
		<rect class="viz-node viz-node--output" x="136" y="320" width="38" height="38"></rect>
		<rect class="viz-node viz-node--focus" x="174" y="320" width="38" height="38" stroke-dasharray="5 3"></rect>
		<rect class="viz-node viz-node--focus" x="212" y="320" width="38" height="38" stroke-dasharray="5 3"></rect>
		<rect class="viz-node" x="250" y="320" width="38" height="38"></rect>
		<rect class="viz-node" x="288" y="320" width="38" height="38"></rect>
		<text class="viz-label" x="193" y="374" text-anchor="middle">boundary could lie here</text>
		<path d="M 326 100 Q 344 100 344 226 L 344 382 Q 344 405 319 405" fill="none" stroke="var(--viz-input-stroke)" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#segmentation-arrow)"></path>
		<text class="viz-axis-label" x="338" y="173" text-anchor="end">SKIP</text>
		<text class="viz-label" x="338" y="188" text-anchor="end">fine edge evidence</text>
		<text class="viz-axis-label" x="22" y="400">4 · FUSE SEMANTICS + SAME-SCALE DETAIL</text>
		<rect class="viz-node viz-node--output" x="22" y="411" width="38" height="20"></rect>
		<rect class="viz-node viz-node--output" x="60" y="411" width="38" height="20"></rect>
		<rect class="viz-node viz-node--output" x="98" y="411" width="38" height="20"></rect>
		<rect class="viz-node viz-node--output" x="136" y="411" width="38" height="20"></rect>
		<rect class="viz-node viz-node--output" x="174" y="411" width="38" height="20"></rect>
		<rect class="viz-node" x="212" y="411" width="38" height="20"></rect>
		<rect class="viz-node" x="250" y="411" width="38" height="20"></rect>
		<rect class="viz-node" x="288" y="411" width="38" height="20"></rect>
		<path d="M 212 408 L 212 434" fill="none" stroke="var(--c-text)" stroke-width="3"></path>
	</svg>
	<figcaption><strong>Read it this way:</strong> start at the fine encoder grid: it retains where the object ends. Downsampling builds stronger context but compresses that edge into a coarse cell. Upsampling can produce an eight-cell mask again, yet the dashed two-cell region shows that a larger canvas does not recreate the discarded location. The dashed skip carries same-scale fine evidence around the bottleneck; fusing it with coarse semantics lets the decoder place the solid boundary. Skips help localization, but they do not guarantee perfect reconstruction. Original synthesis informed by <a href="https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html">Long et al. (2015)</a> and <a href="https://arxiv.org/abs/1505.04597">Ronneberger et al. (2015)</a>.</figcaption>
</figure>

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
