---
title: "Label smoothing"
description: "Replace one-hot targets with a softened distribution that puts ε mass on the wrong classes. Improves calibration, sometimes hurts retrieval."
date: "2026-02-24"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Label smoothing [(Szegedy et al., 2016)](https://arxiv.org/abs/1512.00567) replaces the hard one-hot target $y$ with $\tilde{y} = (1 - \varepsilon) \cdot y + \varepsilon / K$, where $K$ is the number of classes and $\varepsilon$ is a small smoothing constant (typically 0.1). The cross-entropy loss is computed against $\tilde{y}$.

With one-hot targets, cross-entropy approaches its lower bound of zero as the model becomes confident. It can always reduce loss further by making the correct logit larger; reaching the boundary $p_c = 1$ would require an infinite gap between the correct and incorrect logits. This pushes the model toward arbitrarily large logit gaps and overconfident predictions, which are poorly calibrated.

Label smoothing instead creates a finite optimum and forces the model to maintain non-zero probability on incorrect classes. The effects:

- **Better calibration**: predicted probabilities track empirical accuracy more closely.
- **Slightly better generalization** on most classification benchmarks.
- **Standard in transformer training**: original "Attention Is All You Need" used $\varepsilon = 0.1$; LLM pretraining occasionally uses it.

## The mechanism

For a classification problem with $K$ classes and true class $c$:

- Hard target: $y_i = 1$ if $i = c$, else 0.
- Smoothed target: $\tilde{y}_i = 1 - \varepsilon + \varepsilon/K$ if $i = c$, else $\varepsilon/K$.

Cross-entropy with smoothed targets:

$$
L = -\sum_i \tilde{y}_i \log p_i = -(1 - \varepsilon) \log p_c - \frac{\varepsilon}{K} \sum_i \log p_i.
$$

The first term is the standard cross-entropy; the second is an entropy-like penalty that pulls the predicted distribution toward uniform.

Equivalent view: the optimal $p_c$ for label smoothing is $1 - \varepsilon + \varepsilon/K$, not 1. The model has no incentive to push the correct logit beyond what produces this target probability.

<!-- visual:label-smoothing-finite-confidence-optimum -->
<figure class="learning-figure plot-panel" aria-labelledby="label-smoothing-optimum-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="label-smoothing-optimum-title">Uniform mixing leaves target mass 0.92, not 0.90</p><div class="visual-argument" data-visual-argument="label-smoothing-finite-confidence-optimum" data-argument-kind="parts" data-argument-composition="whole-and-share"><svg viewBox="0 0 360 378" role="img" aria-label="Uniform mixing gives the true class part of the uniform mass too. For five classes and smoothing 0.1, the true target is 0.9+0.1/5=0.92. The other four classes receive 0.02 each."><title>Uniform mixing gives the true class part of the uniform mass too</title><desc>For five classes and smoothing 0.1, the true target is 0.9+0.1/5=0.92. The other four classes receive 0.02 each.</desc><text x="20" y="30" style="font-size:22px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">1 target mass</tspan></text><rect data-proof-part="true-class target" data-proof-value="0.92" data-proof-total="1" x="20" y="66" width="294.40000000000003" height="44" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><rect data-proof-part="other four classes combined" data-proof-value="0.08" data-proof-total="1" x="314.40000000000003" y="66" width="25.6" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><text x="20" y="148" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">true-class target</tspan></text><text x="340" y="148" style="font-size:18px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">92.0%</tspan></text><rect x="20" y="176" width="320" height="12" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><rect x="20" y="176" width="294.40000000000003" height="12" style="fill:var(--viz-output-stroke)"></rect><text x="20" y="212" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">0.92 of 1 target mass</tspan></text><text x="20" y="254" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">other four classes</tspan><tspan x="20" dy="18">combined</tspan></text><text x="340" y="254" style="font-size:18px;font-weight:700;text-anchor:end;fill:var(--viz-warning-stroke)"><tspan x="340" dy="0">8.0%</tspan></text><rect x="20" y="300" width="320" height="12" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><rect x="314.40000000000003" y="300" width="25.6" height="12" style="fill:var(--viz-warning-stroke)"></rect><text x="20" y="336" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">0.08 of 1 target mass</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 315" role="img" aria-labelledby="label-smoothing-optimum-svg-title label-smoothing-optimum-svg-desc">
		<title id="label-smoothing-optimum-svg-title">One-hot and label-smoothed cross-entropy near complete confidence</title>
		<desc id="label-smoothing-optimum-svg-desc">An original loss plot for five classes and epsilon 0.1. The horizontal axis shows predicted probability for the correct class from 0.80 to 0.999; the remaining probability is divided equally among four wrong classes. One-hot cross-entropy is a solid curve that keeps falling toward zero as correct-class probability approaches one. Label-smoothed cross-entropy is a dashed curve with a diamond minimum at correct-class probability 0.92, matching the target probabilities 0.92, 0.02, 0.02, 0.02, and 0.02. Beyond 0.92 the smoothed loss rises; at probability 0.99, one-hot loss is 0.010 and smoothed loss is 0.489.</desc>
		<rect class="viz-plot-bg" x="44" y="29" width="301" height="206" rx="4"></rect>
		<path class="viz-gridline" d="M55 35H335 M55 130H335 M55 225H335 M55 35V225 M195.7 35V225 M223.8 35V225 M322.3 35V225"></path>
		<path class="viz-axis" d="M55 35V225H335"></path>
		<path class="viz-roc-curve" d="M55 164.4 L125.4 180.9 L195.7 196.4 L223.8 202.4 L266.1 211.1 L308.3 219.5 L322.3 222.3 L329.4 223.6 L335 224.7"></path>
		<path class="viz-pr-curve" style="stroke-dasharray:7 5" d="M55 104.2 L125.4 113.1 L195.7 118.6 L223.8 119.2 L266.1 117 L308.3 104.9 L322.3 92.4 L329.4 78.6 L335 44.7"></path>
		<path class="viz-operating-guide" d="M223.8 35V225"></path>
		<path class="viz-operating-point" d="M223.8 112.2L230.8 119.2L223.8 126.2L216.8 119.2Z"></path>
		<text class="viz-callout" x="68" y="155">one-hot loss (solid)</text>
		<text class="viz-callout" x="63" y="91">smoothed loss (dashed)</text>
		<text class="viz-callout" x="218" y="79" text-anchor="end">minimum: p<tspan baseline-shift="sub" font-size="8">c</tspan> = 0.92</text>
		<text class="viz-label" x="238" y="141">target = prediction</text>
		<text class="viz-label" x="316" y="67" text-anchor="end">at 0.99:</text>
		<text class="viz-label" x="316" y="81" text-anchor="end">0.010 vs 0.489</text>
		<g class="viz-label" text-anchor="end"><text x="38" y="39">0.70</text><text x="38" y="134">0.35</text><text x="38" y="229">0</text></g>
		<g class="viz-label" text-anchor="middle"><text x="55" y="250">0.80</text><text x="195.7" y="250">0.90</text><text x="223.8" y="250">0.92</text><text x="322.3" y="250">0.99</text></g>
		<text class="viz-axis-label" x="48" y="18">cross-entropy loss</text>
		<text class="viz-axis-label" x="195" y="275" text-anchor="middle">predicted probability for correct class p<tspan baseline-shift="sub" font-size="8">c</tspan></text>
		<text class="viz-label" x="195" y="298" text-anchor="middle">K = 5 · ε = 0.1 · wrong-class mass split equally</text>
	</svg>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> follow the solid curve right: a one-hot target keeps rewarding higher confidence, and its zero-loss boundary requires an unbounded correct-versus-incorrect logit gap. The dashed curve stops at <code>p_c = 0.92</code>, where the prediction matches the smoothed target <code>[0.92, 0.02, 0.02, 0.02, 0.02]</code>. Push farther right and the loss rises because the wrong classes fall below their non-zero targets. This is an original calculation from the formulation in <a href="https://arxiv.org/abs/1512.00567">Szegedy et al. (2016)</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="label-smoothing-finite-confidence-optimum"><strong>Check:</strong> Why is the optimum 0.92, not 0.90, for K=5 and epsilon=0.1 in this convention?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="label-smoothing-finite-confidence-optimum">For five classes and smoothing 0.1, the true target is 0.9+0.1/5=0.92. The other four classes receive 0.02 each.</span></figcaption>
</figure>

## When to use

- **Language modeling or image classification:** compare the declared smoothing convention and strength with unsmoothed training under matched validation.
- **Distillation:** teacher probabilities already supply soft targets; extra smoothing is a separate intervention, not automatically necessary or redundant.
- **Retrieval / contrastive learning:** assess ranking and representation quality under the actual candidate/label semantics instead of transferring a classification default.

## Side effects

- **Calibration and accuracy can change in either direction.** Smoothing changes the training target; it does not prove probability-frequency agreement on deployment data.
- **Representation and distillation effects depend on the task.** Studies such as Müller et al. (2019) motivate checking teacher information and embedding geometry, not declaring one outcome universal.
- **Keep the evaluation target explicit.** Report ordinary held-out likelihood and task metrics separately from the smoothed training objective; see the [cross-entropy mechanism](/concepts/cross-entropy-softmax/).

## Common pitfalls

- **Stacking target changes without testing them.** Mixup/CutMix and smoothing have different constructions; their combination needs an ablation rather than automatic approval or rejection.
- **Using on a regression problem.** Label smoothing is for categorical cross-entropy; it has no meaning for MSE.
- **Treating a coefficient as universal.** Include zero and task-appropriate candidates in validation; the class count and smoothing convention affect what any chosen value means.
- **Forgetting to disable for eval-only metrics.** Loss numbers with label smoothing are not directly comparable to one-hot loss numbers.
