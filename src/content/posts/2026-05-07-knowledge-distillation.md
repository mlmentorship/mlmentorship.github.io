---
title: "Knowledge distillation"
description: "Train a small student to match a large teacher's outputs. The student gets richer signal than from hard labels because the teacher's soft probabilities encode similarity structure."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Knowledge distillation** trains a student model with a loss against a teacher's soft predictions, not the hard label. The student learns the teacher's full output distribution, which carries information about how classes relate ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)).

Hard labels say "this is a 7." Teacher logits say "94 percent 7, 4 percent 1, 1 percent 9, everything else 0.01." That extra structure tells the student that 7 looks more like 1 than like 9. A small model trained against this signal usually beats the same model trained from scratch on hard labels at matched compute.

Distillation is the dominant technique for shrinking large models in production. DistilBERT, TinyBERT, MobileBERT, and most production LLMs ship distilled variants. Often combined with [pruning](/concepts/pruning/) and [quantization](/concepts/quantization/).

## The mechanism

Given teacher logits $z^T$, student logits $z^S$, hard label $y$, temperature $\tau > 1$:

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \text{softmax}(z^S)) + (1 - \alpha) \cdot \tau^2 \cdot \text{KL}\!\left(\text{softmax}(z^T / \tau) \,\|\, \text{softmax}(z^S / \tau)\right).
$$

- **Temperature** $\tau$ softens both distributions. Higher $\tau$ exposes more of the teacher's "dark knowledge" about non-target classes. $\tau = 2$ to $5$ is typical.
- **$\tau^2$ scaling** is needed because softening reduces gradient magnitude by $1/\tau^2$.
- **$\alpha$** weights the hard-label loss. $\alpha = 0$ gives pure distillation; $\alpha \in [0.1, 0.5]$ is common.

<!-- visual:distillation-temperature-dark-knowledge -->
<figure class="learning-figure plot-panel" aria-labelledby="distillation-temperature-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="distillation-temperature-title">Higher temperature exposes alternatives without changing the winning class</p><div class="visual-argument" data-visual-argument="distillation-temperature-dark-knowledge" data-argument-kind="matrix" data-argument-composition="soften-mass"><svg viewBox="0 0 360 601" role="img" aria-label="Temperature softens alternatives without changing the winner. For logits (2,1,0), temperature 2 increases the probability of both losing classes. The top class remains the same."><title>Temperature softens alternatives without changing the winner</title><desc>For logits (2,1,0), temperature 2 increases the probability of both losing classes. The top class remains the same.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Heating redistributes mass without</tspan><tspan x="20" dy="18">changing order</tspan></text><text x="20" y="100" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Temperature 1: same three classes</tspan></text><rect data-picture-softened-class="0" data-picture-temperature="1" data-picture-probability="0.6652409557748219" x="20" y="151" width="212.877105847943" height="55" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="256" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A 0.6652</tspan></text><rect data-picture-softened-class="1" data-picture-temperature="1" data-picture-probability="0.24472847105479764" x="232.877105847943" y="151" width="78.31311073753524" height="55" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="131" y="256" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="131" dy="0">B 0.2447</tspan></text><rect data-picture-softened-class="2" data-picture-temperature="1" data-picture-probability="0.09003057317038046" x="311.19021658547825" y="151" width="28.809783414521746" height="55" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="242" y="256" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="242" dy="0">C 0.09003</tspan></text><text x="20" y="327" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Temperature 2: same three classes</tspan></text><rect data-picture-softened-class="0" data-picture-temperature="2" data-picture-probability="0.506480391055654" x="20" y="378" width="162.07372513780928" height="55" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="483" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A 0.5065</tspan></text><rect data-picture-softened-class="1" data-picture-temperature="2" data-picture-probability="0.3071958857184984" x="182.07372513780928" y="378" width="98.30268342991948" height="55" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="131" y="483" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="131" dy="0">B 0.3072</tspan></text><rect data-picture-softened-class="2" data-picture-temperature="2" data-picture-probability="0.18632372322584756" x="280.37640856772873" y="378" width="59.62359143227122" height="55" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="242" y="483" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="242" dy="0">C 0.1863</tspan></text><text x="20" y="554" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">A stays largest; B and C gain</tspan><tspan x="20" dy="18">probability.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 350" role="img" aria-labelledby="distillation-temperature-svg-title distillation-temperature-svg-desc">
		<title id="distillation-temperature-svg-title">Hard and soft targets for a teacher classifying a handwritten seven</title>
		<desc id="distillation-temperature-svg-desc">Three aligned horizontal bar groups compare probabilities for classes 7, 1, and 9. The hard label is 100, 0, and 0 percent. Teacher logits 5, 2, and 1 produce 93.6, 4.7, and 1.7 percent at temperature 1. The same logits at temperature 3 produce 61.3, 22.5, and 16.2 percent. Raising temperature preserves the ordering 7 above 1 above 9 while exposing the teacher's relative preference for the two wrong classes. The student is trained to match this softened distribution.</desc>
		<g class="viz-axis-label" text-anchor="middle"><text x="141" y="18">class 7</text><text x="233" y="18">class 1</text><text x="325" y="18">class 9</text></g>
		<rect class="viz-plot-bg" x="4" y="29" width="352" height="82" rx="5"></rect>
		<text class="viz-callout" x="12" y="51">Hard label</text>
		<text class="viz-label" x="12" y="68">one-hot target</text>
		<g class="viz-node viz-node--focus"><rect x="100" y="41" width="82" height="24" rx="3"></rect><rect x="192" y="63" width="0" height="24" rx="3"></rect><rect x="284" y="63" width="0" height="24" rx="3"></rect></g>
		<g class="viz-callout" text-anchor="middle"><text x="141" y="57">100%</text><text x="233" y="82">0%</text><text x="325" y="82">0%</text></g>
		<text class="viz-label" x="100" y="101">Only “this is 7”</text>
		<rect class="viz-plot-bg" x="4" y="121" width="352" height="92" rx="5"></rect>
		<text class="viz-callout" x="12" y="143">Teacher · τ = 1</text>
		<text class="viz-label" x="12" y="160">logits [5, 2, 1]</text>
		<g class="viz-node viz-node--input"><rect x="100" y="137" width="77" height="24" rx="3"></rect><rect x="192" y="161" width="4" height="24" rx="2"></rect><rect x="284" y="161" width="2" height="24" rx="1"></rect></g>
		<g class="viz-callout" text-anchor="middle"><text x="138.5" y="153">93.6%</text><text x="233" y="180">4.7%</text><text x="325" y="180">1.7%</text></g>
		<text class="viz-label" x="100" y="203">Non-target preferences are present but tiny</text>
		<rect class="viz-plot-bg" x="4" y="223" width="352" height="119" rx="5"></rect>
		<text class="viz-callout" x="12" y="245">Transfer target · τ = 3</text>
		<text class="viz-label" x="12" y="262">same teacher logits</text>
		<g class="viz-node viz-node--output"><rect x="100" y="241" width="50" height="24" rx="3"></rect><rect x="192" y="270" width="18" height="24" rx="3"></rect><rect x="284" y="270" width="13" height="24" rx="3"></rect></g>
		<g class="viz-callout" text-anchor="middle"><text x="125" y="257">61.3%</text><text x="233" y="289">22.5%</text><text x="325" y="289">16.2%</text></g>
		<path class="viz-operating-guide" d="M210 304H297"></path>
		<text class="viz-callout" x="253.5" y="319" text-anchor="middle">1 remains above 9</text>
		<text class="viz-label" x="100" y="334">Student matches this distribution at τ = 3</text>
	</svg>
	<div class="visual-derivation-caption">compare each class vertically. Dividing the fixed teacher logits <code>[5, 2, 1]</code> by a higher temperature moves probability away from class 7, but it does not change the ranking: class 1 remains more plausible than class 9. The softened <code>22.5%</code> versus <code>16.2%</code> target gives the student similarity information that the one-hot label discards. Teacher and student use the same temperature for the KL term; the separate hard-label term still points to class 7. Values are an original calculation following <a href="https://arxiv.org/abs/1503.02531">Hinton et al. (2015)</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="distillation-temperature-dark-knowledge"><strong>Check:</strong> What changes between the two teacher distributions? What preference remains unchanged?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="distillation-temperature-dark-knowledge">For logits (2,1,0), temperature 2 increases the probability of both losing classes. The top class remains the same. <br><strong>Does heating these logits change which class is largest?</strong></span></figcaption>
</figure>

## Variants

| Variant | What it matches |
|---|---|
| **Logit distillation** (above) | Teacher output logits |
| **Feature distillation** ([FitNets](https://arxiv.org/abs/1412.6550)) | Intermediate hidden states |
| **Attention distillation** ([TinyBERT](https://arxiv.org/abs/1909.10351)) | Teacher attention maps |
| **Sequence-level distillation** ([Kim & Rush, 2016](https://arxiv.org/abs/1606.07947)) | Teacher's most likely outputs (for autoregressive models) |
| **Self-distillation** | Teacher and student are the same architecture; sometimes the teacher is a previous training checkpoint |

For LLMs, sequence-level distillation against teacher samples (or rejection-sampled teacher outputs) is the dominant recipe. Logit distillation is impractical at vocab size 100k+.

## When it works and when it doesn't

Works well when:

- Teacher is significantly better than what the student could reach alone.
- Student capacity is at least 10 to 20 percent of the teacher.
- Training data overlaps the teacher's training distribution.

Fails when:

- Student is too small. Capacity gap is the dominant ceiling.
- Teacher is already small. The "dark knowledge" margin is thin.
- Distribution shift. Teacher predictions are unreliable on student's deployment data.

## Common pitfalls

- **Forgetting $\tau^2$ scaling.** Without it, the KL term has tiny gradients and the hard-label term dominates.
- **Distilling only logits when feature distillation would help.** For very small students, intermediate matching is often required.
- **Skipping the temperature.** $\tau = 1$ collapses the teacher's distribution to nearly one-hot for confident predictions; you lose most of the signal.
- **Training student on teacher-correct examples only.** The interesting signal is on examples where the teacher is uncertain. Use the full training set.

## Related

- [Pruning](/concepts/pruning/).
- [Quantization](/concepts/quantization/).
- [Cross-entropy loss](/concepts/cross-entropy-softmax/).
