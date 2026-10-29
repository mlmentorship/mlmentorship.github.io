---
title: "Mixup and CutMix"
description: "Two data-augmentation schemes that train on convex combinations of pairs of inputs and their labels. Strong regularization for image classification; sometimes used in audio and tabular."
date: "2025-10-17"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Mixup blends paired inputs and class targets with the same coefficient; CutMix replaces a region and uses realized-area target weights.
The constructions impose training conventions on synthetic inputs, not a guarantee that semantic evidence is proportional to those coefficients.
Quality, calibration, and robustness effects require evaluation under the actual task and recipe.

## Mechanism

### Mixup

For paired same-shaped numeric inputs and class-probability targets, choose $0\le\lambda\le1$:

$$
\tilde x = \lambda x_i + (1 - \lambda) x_j, \quad \tilde y = \lambda y_i + (1 - \lambda) y_j.
$$

With $x_A=(0,4)$, $x_B=(4,0)$, labels A/B, and $\lambda=0.75$, the input becomes $(1,3)$.
The target is $q=(0.75,0.25)$, a nonnegative distribution summing to one.
At illustrative mixed-input prediction $(0.6,0.4)$, soft-target cross-entropy is:

$$
-0.75\log(0.6)-0.25\log(0.4)\approx0.612191901.
$$

The weighted class losses must use this same mixed-input prediction, not predictions on the two original inputs.
For a batch, preserve the example permutation and align every input/target coefficient; declare per-batch or per-example sampling.

### CutMix

Let binary M equal one on retained A pixels and zero on the pasted B region:

$$
{}\tilde x=Mx_A+(1-M)x_B,\qquad
\lambda_{real}=\frac{\sum M}{HW},\qquad
q=\lambda_{real}y_A+(1-\lambda_{real})y_B.
$$

Broadcast the spatial mask across image channels.
The pictured four-cell patch in a $4\times4$ grid gives $\lambda_{real}=12/16=0.75$ and the same target as above.
The actual synthetic input differs: pasted regions are not pixelwise blends.

<!-- visual:mixup-cutmix-input-target-coupling -->
<figure class="learning-figure plot-panel" aria-labelledby="mixup-cutmix-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="mixup-cutmix-visual-title">The realized patch area determines the mixed target weights</p>
	<svg viewBox="0 0 360 550" role="img" aria-labelledby="mixup-cutmix-svg-title mixup-cutmix-svg-desc">
		<title id="mixup-cutmix-svg-title">Mixup and CutMix input-target coupling with lambda equal to 0.75</title>
		<desc id="mixup-cutmix-svg-desc">Source A is represented by horizontal stripes and class label y A. Source B is represented by dots and class label y B. For Mixup, every input location contains 75 percent A and 25 percent B, and the soft target is 0.75 y A plus 0.25 y B. For CutMix, a 2 by 2 dotted B patch replaces part of a 4 by 4 striped A grid. Twelve of sixteen cells remain A and four of sixteen are B, so the realized retained-area lambda is 12 over 16, or 0.75, and the soft target is again 0.75 y A plus 0.25 y B.</desc>
		<defs>
			<pattern id="mix-source-a" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0 4H8" style="stroke:var(--viz-input-stroke);stroke-width:2"></path></pattern>
			<pattern id="mix-source-b" width="8" height="8" patternUnits="userSpaceOnUse"><circle cx="4" cy="4" r="1.6" style="fill:var(--viz-focus-stroke)"></circle></pattern>
		</defs>
		<text class="viz-axis-label" x="20" y="18">PAIRED SOURCES</text>
		<rect class="viz-node viz-node--input" x="20" y="30" width="130" height="62" rx="4"></rect><rect x="28" y="38" width="52" height="46" rx="2" style="fill:url(#mix-source-a);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text class="viz-node-label" x="112" y="57">A</text><text class="viz-node-value" x="112" y="76">label y<tspan baseline-shift="sub" font-size="8">A</tspan></text>
		<rect class="viz-node viz-node--focus" x="210" y="30" width="130" height="62" rx="4"></rect><rect x="218" y="38" width="52" height="46" rx="2" style="fill:url(#mix-source-b);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text class="viz-node-label" x="302" y="57">B</text><text class="viz-node-value" x="302" y="76">label y<tspan baseline-shift="sub" font-size="8">B</tspan></text>
		<text class="viz-callout" x="20" y="130">MIXUP · λ = 0.75</text><text class="viz-edge-label" x="340" y="130" style="text-anchor:end">blend every location</text>
		<rect class="viz-node" x="20" y="144" width="140" height="100" rx="4"></rect><rect x="28" y="152" width="124" height="64" rx="2" style="fill:var(--viz-input-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><rect x="28" y="152" width="124" height="64" rx="2" style="fill:url(#mix-source-a)"></rect><rect x="28" y="152" width="124" height="64" rx="2" style="fill:url(#mix-source-b);opacity:.55"></rect><text class="viz-node-value" x="90" y="234">A + B at every coordinate</text>
		<path class="viz-axis" d="M160 194H180"></path><path class="viz-arrow-forward" d="M186 194l-9-5v10Z"></path>
		<rect class="viz-node viz-node--output" x="186" y="144" width="154" height="100" rx="4"></rect><text class="viz-node-value" x="263" y="166">SOFT TARGET</text><rect x="198" y="178" width="96" height="24" rx="2" style="fill:url(#mix-source-a);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="294" y="178" width="32" height="24" rx="2" style="fill:url(#mix-source-b);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text class="viz-edge-label" x="246" y="195">75% A</text><text class="viz-edge-label" x="310" y="195">25% B</text><text class="viz-node-value" x="263" y="226">0.75 y<tspan baseline-shift="sub" font-size="8">A</tspan> + 0.25 y<tspan baseline-shift="sub" font-size="8">B</tspan></text>
		<text class="viz-callout" x="20" y="286">CUTMIX · REALIZED λ = 12/16 = 0.75</text><text class="viz-edge-label" x="340" y="304" style="text-anchor:end">replace a countable region</text>
		<rect class="viz-node" x="20" y="316" width="140" height="170" rx="4"></rect><rect x="28" y="324" width="124" height="124" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="28" y="324" width="124" height="124" style="fill:url(#mix-source-a)"></rect><rect x="90" y="386" width="62" height="62" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:3"></rect><rect x="90" y="386" width="62" height="62" style="fill:url(#mix-source-b)"></rect><path d="M59 324V448M90 324V448M121 324V448M28 355H152M28 386H152M28 417H152" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1"></path><text class="viz-node-value" x="90" y="467">12 striped A cells</text><text class="viz-node-value" x="90" y="480">4 dotted B cells</text>
		<path class="viz-axis" d="M160 386H180"></path><path class="viz-arrow-forward" d="M186 386l-9-5v10Z"></path>
		<rect class="viz-node viz-node--output" x="186" y="336" width="154" height="130" rx="4"></rect><text class="viz-node-value" x="263" y="358">AREA-MATCHED TARGET</text><rect x="198" y="370" width="96" height="24" rx="2" style="fill:url(#mix-source-a);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect x="294" y="370" width="32" height="24" rx="2" style="fill:url(#mix-source-b);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text class="viz-edge-label" x="246" y="387">12/16 A</text><text class="viz-edge-label" x="310" y="387">4/16 B</text><text class="viz-node-value" x="263" y="416">λ = retained A area</text><text class="viz-node-value" x="263" y="436">0.75 y<tspan baseline-shift="sub" font-size="8">A</tspan> + 0.25 y<tspan baseline-shift="sub" font-size="8">B</tspan></text><text class="viz-edge-label" x="263" y="454">hard boundary · soft label</text>
		<rect class="viz-node viz-node--focus" x="20" y="510" width="320" height="28" rx="4"></rect><text class="viz-callout" x="180" y="529" text-anchor="middle">input contribution = target contribution</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> Mixup blends every coordinate using 75/25 coefficients. CutMix pastes four of sixteen cells, leaving twelve A cells. Both targets are 75/25 here, although the inputs differ. Original schematic follows the <a href="https://arxiv.org/abs/1710.09412">Mixup paper</a>, <a href="https://arxiv.org/abs/1905.04899">CutMix paper</a>, and <a href="https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html">Torchvision guide</a>. <span class="visual-reader-check" data-reader-check="mixup-cutmix-input-target-coupling"><strong>Check:</strong> Why must the CutMix target use the realized 12-of-16 retained cells rather than an unadjusted requested patch coefficient?</span></figcaption>
</figure>

### Change the Interpolation Coefficient

**New task:** Set lambda to one-half for the same paired vectors.

**Picture and change:** Change both input and target: $\tilde x=(2,2)$ and $q=(0.5,0.5)$.

**Check:** Holding the illustrative prediction at $(0.6,0.4)$ gives loss $0.713558178$.
The actual model's new prediction can differ; retaining the old target would mismatch the specified construction.

**Use when:** The sampled coefficient changes; alpha controls its distribution, not a deterministic mixing result.

### Clip a Boundary-Crossing Rectangle

**New task:** A requested $2\times2$ patch leaves only $1\times2$ inside the $4\times4$ image.

**Picture and change:** Use half-open coordinates, clip ordered rectangle bounds to the image, and count actual replaced pixels.
For $x\in[-1,1)$ and $y\in[1,3)$, clipped width is one and height two.

**Check:** Two inserted cells give $\lambda_{real}=14/16=0.875$ and target $(0.875,0.125)$, not $(0.75,0.25)$.

**Use when:** Boundaries or integer rounding change realized area.
An area-weighted target is still a convention: that small patch could remove the only visible A object.

<!-- learning:core-end -->
<div data-learning-extension>

## Choosing $\alpha$

For $\lambda\sim Beta(\alpha,\alpha)$ with positive alpha, the mean is one-half and variance is $1/[4(2\alpha+1)]$.
Small alpha favors endpoints; large alpha concentrates near one-half.
The original [Mixup](https://arxiv.org/abs/1710.09412) and [CutMix](https://arxiv.org/abs/1905.04899) papers report recipe-specific experiments, not universal improvement magnitudes.

## Why it works (intuition)

[Vicinal risk minimization](https://papers.nips.cc/paper/2000/hash/ba9a56ce0a9bfa26e8ed9e10b2cc8f46-Abstract.html) motivates training on chosen neighborhoods around examples.
Those neighborhoods encode assumptions; synthetic points need not belong to the deployment distribution or preserve its correct labels.

## Common pitfalls

- **Label-only mixing:** it is not the stated paired-input augmentation, whatever separate objective it implements.
- **Wrong target unit:** image classification, segmentation, boxes, and sequences require task-specific target handling.
- **Evaluation contamination:** disable random training mixing for ordinary clean-example evaluation.
- **Recipe interactions:** augmentation and target smoothing can jointly overregularize; evaluate matched ablations.

## Related

- [Label smoothing](/concepts/label-smoothing/). Another way to soften targets.
- [Dropout](/concepts/dropout/). Stochastic activation regularization.
- [Regularization](/concepts/regularization/). Overview.

</div>
