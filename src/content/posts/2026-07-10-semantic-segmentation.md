---
title: "Semantic segmentation"
description: "Assign a class to every pixel: encoder-decoder architectures, losses, IoU, class imbalance, boundaries, and deployment constraints."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Semantic segmentation predicts a semantic class at each evaluated pixel, rather than a box or a separate object identity.
An encoder can recognize an object while discarding its precise boundary.
Ask which spatial evidence survives and which denominator reveals a small-class failure.

Output logits have shape $B\times C\times H\times W$: batch, classes, height, and width.
Align predictions with the target coordinates and exclude declared ignore labels.
Resize categorical targets by nearest-neighbor unless a separate soft-label policy applies.

## Architectures

- **FCN:** replaces dense heads with convolutional prediction.
- **U-Net:** combines coarse semantics with retained higher-resolution encoder features.
- **DeepLab:** uses dilation and multiscale context with a declared output stride.
- **Transformer decoders:** combine global context with learned masks or pixel queries.

If two boundaries produce the same coarse representation, its decoder alone cannot identify which boundary occurred.
For example, averaging $(1,0)$ and $(0,1)$ yields the same value $1/2$.
Interpolation restores size, not that erased distinction.
A learned prior may infer plausible detail; a skip supplies additional evidence without guaranteeing exact recovery.

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
	<figcaption><strong>Read it this way:</strong> the fine grid retains the edge, while the coarse grid leaves an uncertain cell. Enlarging that cell does not restore distinctions already erased. The dashed skip supplies fine evidence that can distinguish candidate boundaries; success still depends on those features and the decoder. The illustrated sharp edge is an outcome, not a reconstruction guarantee. Original synthesis informed by <a href="https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html">Long et al. (2015)</a> and <a href="https://arxiv.org/abs/1505.04597">Ronneberger et al. (2015)</a>.</figcaption>
</figure>

## Losses

Mean pixel cross-entropy averages only evaluated pixels; an empty valid set needs an explicit skip/error policy.
Weighted or focal losses change contributions; Dice emphasizes overlap; boundary terms emphasize localization.
None removes annotation uncertainty or automatically supplies calibrated probabilities.

## Metrics

Intersection over Union for class $c$ is

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}.$$

For 95 background and five object pixels, predicting background everywhere gives:

| Class | TP | FP | FN | IoU |
| --- | --- | --- | --- | --- |
| Object | 0 | 0 | 5 | 0 |
| Background | 95 | 5 | 0 | 0.95 |

Accuracy is $0.95$, while two-class macro mIoU is $(0+0.95)/2=0.475$.
The metric correctly measures pixel agreement, not rare-class success.
Zero IoU denominators require a stated absent-class policy; ignored pixels are not background.

### Make rare positives more important

**New task:** prioritize the small foreground class.

**Picture and change:** weight positive cross-entropy terms by $19$, background terms by $1$.

**Check:** weighted class masses become $19(5)=95$ and $1(95)=95$.
A constant-score model's optimum changes from foreground probability $0.05$ to weighted score $0.5$.
Report class-specific results and validate calibration rather than interpreting the weighted score as the original probability.

**Use when:** missed foreground carries different task costs; the all-background object's IoU remains zero.

### Fit inference into smaller tiles

**New task:** full-resolution activations exceed memory.

**Picture and change:** a $512^2$ tile uses 262,144 pixels versus 921,600 for a $960^2$ image.
Along each axis, origins $0$ and $448$ produce a 64-pixel overlap.

**Check:** global coordinate $500$ is local coordinate $52$ in the second tile.
Align and blend overlapping logits before class selection; do not average discrete class IDs.
Context, padding, and tile boundaries may still differ from full-image inference.

**Use when:** activation memory requires tiling, with measured overlap, latency, and boundary checks.

<!-- learning:core-end -->

<div data-learning-extension>

## In an interview

Define the spatial transform, ignore mask, label set, and evaluation unit before choosing an architecture.
Report per-class IoU, the declared class average, boundary quality, and relevant object/task outcomes.
Memory savings require accounting for feature channels, temporary workspaces, tiling overlap, and batch size, not pixels alone.

## Common confusions

Hard binary Dice is $2TP/(2TP+FP+FN)$.
One soft convention uses $(2\sum_u p_uy_u+\epsilon)/(\sum_u p_u+\sum_u y_u+\epsilon)$.
Declare $\epsilon$, valid pixels, per-image versus pooled reduction, and the empty-mask policy.
Other denominator conventions define different objectives.

Boundary penalties can emphasize uncertain or inconsistent annotations.
Confidence thresholds and abstention need validation on the deployment population; higher mask precision is not automatically higher task value.

Can semantic labels distinguish two touching objects of the same class?

<details><summary>Answer</summary>

Not necessarily: both objects can share the same semantic label throughout their union.
Instance segmentation supplies separate identities; detection supplies boxes rather than full pixel assignments.

</details>

</div>

*Related: [CNN architecture](/concepts/cnn-architecture/), [ResNet](/concepts/resnet/), and [object detection](/concepts/object-detection-overview/).*
