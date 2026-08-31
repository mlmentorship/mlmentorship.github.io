---
title: "Mixup and CutMix"
description: "Two data-augmentation schemes that train on convex combinations of pairs of inputs and their labels. Strong regularization for image classification; sometimes used in audio and tabular."
date: "2025-10-17"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Mixup** [(Zhang et al., 2018)](https://arxiv.org/abs/1710.09412) trains the model on convex combinations of pairs of training examples: $\tilde x = \lambda x_i + (1 - \lambda) x_j$ and $\tilde y = \lambda y_i + (1 - \lambda) y_j$ with $\lambda \sim \text{Beta}(\alpha, \alpha)$. **CutMix** [(Yun et al., 2019)](https://arxiv.org/abs/1905.04899) instead pastes a rectangular patch from $x_j$ onto $x_i$ and mixes labels by the area ratio.

Both techniques regularize by training on examples between the original training points. Empirically:

- Improve top-1 accuracy on ImageNet by ~1–2% over baseline.
- Improve calibration (predicted probabilities track accuracy better).
- Improve robustness to label noise and adversarial perturbations.
- Standard in modern image classification recipes (timm, ConvNeXt, ViT-style training).

Less common in NLP (token mixing is non-trivial) and in pretraining (large data already covers the input space well). Sometimes used in audio (mix waveforms or spectrograms) and tabular (interpolate features).

## Mechanism

### Mixup

For each batch, sample $\lambda \sim \text{Beta}(\alpha, \alpha)$ once (or per sample) with $\alpha$ small (typical 0.2–0.4). For paired examples $(x_i, y_i)$ and $(x_j, y_j)$:

$$
\tilde x = \lambda x_i + (1 - \lambda) x_j, \quad \tilde y = \lambda y_i + (1 - \lambda) y_j.
$$

Train normally on $(\tilde x, \tilde y)$ with cross-entropy. The label is a soft target.

### CutMix

Sample $\lambda \sim \text{Beta}(\alpha, \alpha)$. Pick a random rectangle in $x_i$ of area $1 - \lambda$ (e.g., width and height $\sqrt{1 - \lambda}$ times the image). Paste the corresponding region from $x_j$ into $x_i$. Mix labels by the area ratio $\lambda$.

The resulting image has a clear local boundary (no blending). Models trained on CutMix often produce more localized class activations.

<!-- visual:mixup-cutmix-input-target-coupling -->
<figure class="learning-figure plot-panel" aria-labelledby="mixup-cutmix-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="mixup-cutmix-visual-title">Match each input-mixing operation to the same weights in its soft target.</p>
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
	<figcaption><strong>Read it this way:</strong> keep the paired sources and target weights fixed. Mixup applies the 75/25 coefficient at every input coordinate. CutMix does no pixel blending: the dotted source B occupies 4 of 16 cells, so source A retains 12/16 of the area and receives 75% of the target. In real code, use the patch area that remains after image-boundary clipping. Original schematic checked against the <a href="https://arxiv.org/abs/1710.09412">Mixup paper</a>, the <a href="https://arxiv.org/abs/1905.04899">CutMix paper</a>, and the <a href="https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html">Torchvision guide</a>.</figcaption>
</figure>

## Choosing $\alpha$

| Setting | Mixup $\alpha$ | CutMix $\alpha$ |
|---------|---------------|----------------|
| ImageNet from scratch | 0.2 | 1.0 |
| Small datasets | 0.2 (more aggressive Mixup hurts) | 1.0 |
| ViT training | 0.2 + CutMix 1.0 (used together) |. |

$\alpha \to 0$ gives near-original samples (almost no mixing); $\alpha \to \infty$ gives $\lambda \approx 0.5$ (always equally mixed). $\alpha$ between 0.2 and 1 is the empirical sweet spot.

## Why it works (intuition)

- **Vicinal risk minimization** [(Chapelle et al., 2001)](https://papers.nips.cc/paper/2000/hash/ba9a56ce0a9bfa26e8ed9e10b2cc8f46-Abstract.html): training on a vicinity around each point regularizes the decision boundary.
- **Empirically**: smoother decision functions, better calibration, less overconfidence on out-of-distribution inputs.
- **Equivalent to** an implicit form of weight regularization in the linear case.

## Common pitfalls

- **Mixing labels but not inputs.** Some implementations mix targets without mixing inputs; this is just label noise, not Mixup.
- **Combining with strong cropping.** Mixup + RandomResizedCrop + label smoothing + AutoAugment is the modern recipe but can over-regularize small datasets.
- **Using on detection / segmentation directly.** Class labels mix easily; bounding boxes do not. Variants like Mosaic (YOLOv4) handle this.
- **Forgetting to disable for evaluation.** Eval should use clean images.

## Related

- [Label smoothing](/concepts/label-smoothing/). Another way to soften targets.
- [Dropout](/concepts/dropout/). Stochastic activation regularization.
- [Regularization](/concepts/regularization/). Overview.
