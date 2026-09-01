---
title: "Multi-task learning and objective interference"
description: "Shared training can improve data efficiency or cause negative transfer. Diagnose task balance through labels, loss scales, gradients, calibration, and per-task outcomes."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["multi-task learning", "multitask learning", "negative transfer", "gradient conflict", "MMoE", "PLE", "GradNorm"]
roles: ["Applied Scientist", "Ranking MLE", "Multimodal ML", "Machine Learning Engineer"]
rounds: ["ML breadth", "ML system design", "Training"]
difficulty: "Intermediate"
priority: "Role-specific"
prerequisites: ["neural-network-training-recipe", "calibration"]
---

## Summary

Multi-task learning trains one model on several related targets. Shared representations can improve data efficiency and regularize sparse tasks. They can also create negative transfer when one task dominates updates or needs features that conflict with another task.

A sound design defines the primary outcome, masks missing labels, normalizes loss scales, checks gradient interaction, and evaluates every task separately. Architecture and loss weights should follow measured task relationships.

**Learning objective:** distinguish task-gradient magnitude imbalance from directional conflict so the diagnosis, rather than equal raw loss weights, determines the intervention.

## Basic objective

For tasks $t=1,\ldots,T$, a common objective is

$$
L(\theta)=\sum_{t=1}^{T} w_t L_t(\theta).
$$

The weights $w_t$ define the optimization tradeoff. They are not product weights unless each loss has the same scale and meaning.

One task may have millions of labels while another has thousands. Cross-entropy, squared error, and ranking losses can also have different numeric scales. A weight of 1 for every task does not give every task equal influence.

## Why shared training can help

Related tasks can provide:

- more supervision for a shared representation;
- useful auxiliary labels when the primary label is sparse;
- regularization against overfitting one target;
- features that transfer across tasks;
- one serving pass for several outputs.

For recommendation, clicks provide dense but biased signal. Surveys provide sparse satisfaction signal. Joint training can use both, provided the dense task does not erase the sparse one.

## Hard and soft sharing

**Hard sharing** uses one trunk with task-specific heads. It is parameter-efficient and common in production.

```text
features -> shared trunk -> task A head
                         -> task B head
```

Hard sharing assumes that a common representation helps every task. It can fail when tasks need different features or have incompatible label processes.

**Soft sharing** gives each task a separate model and adds a penalty or exchange mechanism that encourages shared structure. It costs more but gives tasks more independence.

Start with hard sharing when tasks are related and serving cost matters. Split layers or models only after measuring interference.

## Negative transfer

Negative transfer occurs when adding a task harms another task that matters.

Let $g_a=\nabla_\theta L_a$ and $g_b=\nabla_\theta L_b$. If

$$
g_a^\top g_b < 0,
$$

the two gradients disagree locally. Improving one loss along its gradient can worsen the other.

Gradient conflict is one diagnostic, not a complete explanation. Tasks may conflict only in some layers, examples, or training phases. Data imbalance, bad labels, and different convergence rates can produce similar symptoms.

Track per-task validation metrics before and after adding each task.

<!-- visual:multitask-gradient-diagnosis -->
<figure class="learning-figure" aria-labelledby="multitask-gradient-title">
	<p class="visual-kicker">Two diagnostics, two problems</p>
	<p class="visual-title" id="multitask-gradient-title">Gradient size asks who dominates; the dot product asks whether the tasks disagree.</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison of task-gradient magnitude imbalance and directional conflict">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 330" role="img" aria-labelledby="gradient-scale-title gradient-scale-desc">
				<title id="gradient-scale-title">Unequal but aligned task gradients</title>
				<desc id="gradient-scale-desc">At the same shared-parameter point, task A has gradient vector four comma zero with norm four, while task B has vector one comma zero with norm one. Both point right, their dot product is positive four, and their unweighted sum is five comma zero. Task A supplies four fifths of the sum, so this is magnitude dominance without directional conflict.</desc>
				<defs><marker id="gradient-scale-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
				<text class="viz-axis-label" x="10" y="17">CASE 1 - MAGNITUDE IMBALANCE</text>
				<rect class="viz-plot-bg" x="8" y="28" width="284" height="205" rx="5"></rect>
				<path d="M50 145H268M50 200V55" style="fill:none;stroke:var(--viz-edge);stroke-width:1"></path>
				<text class="viz-edge-label" x="270" y="159">θ₁</text>
				<text class="viz-edge-label" x="39" y="58">θ₂</text>
				<circle cx="50" cy="145" r="4" style="fill:var(--c-text-soft)"></circle>
				<path d="M50 145H250" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;marker-end:url(#gradient-scale-arrow)"></path>
				<path d="M50 145H100" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:4;stroke-dasharray:5 3;marker-end:url(#gradient-scale-arrow)"></path>
				<path d="M250 137V153M242 145H258" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<rect x="96" y="141" width="8" height="8" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
				<text class="viz-callout" x="190" y="116" text-anchor="middle">g_A = (4, 0), ‖g_A‖ = 4</text>
				<text class="viz-callout" x="103" y="178" text-anchor="middle">g_B = (1, 0), ‖g_B‖ = 1</text>
				<text class="viz-axis-label" x="150" y="220" text-anchor="middle">SAME DIRECTION - DIFFERENT LENGTH</text>
				<rect class="viz-node viz-node--output" x="18" y="247" width="264" height="67" rx="4"></rect>
				<text class="viz-callout" x="150" y="268" text-anchor="middle">g_A · g_B = 4 &gt; 0 (aligned)</text>
				<text class="viz-callout" x="150" y="288" text-anchor="middle">g_A + g_B = (5, 0); A supplies 80%</text>
				<text class="viz-axis-label" x="150" y="306" text-anchor="middle">DIAGNOSIS: SCALE ISSUE, NOT CONFLICT</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 330" role="img" aria-labelledby="gradient-conflict-title gradient-conflict-desc">
				<title id="gradient-conflict-title">Equal-sized but conflicting task gradients</title>
				<desc id="gradient-conflict-desc">At the same shared-parameter point, task A has gradient three comma one and task B has gradient negative three comma one. Both norms equal square root of ten, but their dot product is negative eight. Their horizontal components cancel, producing a sum of zero comma two. Equalizing gradient norms cannot remove this directional conflict.</desc>
				<defs><marker id="gradient-conflict-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
				<text class="viz-axis-label" x="10" y="17">CASE 2 - DIRECTIONAL CONFLICT</text>
				<rect class="viz-plot-bg" x="8" y="28" width="284" height="205" rx="5"></rect>
				<path d="M28 175H272M150 213V48" style="fill:none;stroke:var(--viz-edge);stroke-width:1"></path>
				<text class="viz-edge-label" x="274" y="189">θ₁</text>
				<text class="viz-edge-label" x="139" y="51">θ₂</text>
				<circle cx="150" cy="175" r="4" style="fill:var(--c-text-soft)"></circle>
				<path d="M150 175L240 145" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;marker-end:url(#gradient-conflict-arrow)"></path>
				<path d="M150 175L60 145" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-dasharray:6 3;marker-end:url(#gradient-conflict-arrow)"></path>
				<path d="M150 175V115" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:4;marker-end:url(#gradient-conflict-arrow)"></path>
				<path d="M240 137V153M232 145H248" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<rect x="56" y="141" width="8" height="8" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect>
				<text class="viz-callout" x="224" y="128" text-anchor="middle">g_A = (3, 1)</text>
				<text class="viz-callout" x="76" y="128" text-anchor="middle">g_B = (-3, 1)</text>
				<text class="viz-callout" x="150" y="94" text-anchor="middle">sum = (0, 2)</text>
				<text class="viz-axis-label" x="150" y="220" text-anchor="middle">EQUAL LENGTH - OPPOSING COMPONENTS</text>
				<rect class="viz-node" x="18" y="247" width="264" height="67" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect>
				<text class="viz-callout" x="150" y="268" text-anchor="middle">‖g_A‖ = ‖g_B‖ = √10</text>
				<text class="viz-callout" x="150" y="288" text-anchor="middle">g_A · g_B = -8 &lt; 0 (conflict)</text>
				<text class="viz-axis-label" x="150" y="306" text-anchor="middle">DIAGNOSIS: DIRECTION ISSUE, NOT SCALE</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare lengths first, then angles. On the left, task A dominates the unweighted sum, but both gradients favor the same local direction; normalize or reweight only if that dominance is unintended. On the right, the norms are already equal, yet the negative dot product means a step down one task's gradient raises the other loss to first order. Norm balancing alone cannot remove that conflict: evaluate task outcomes, then consider less sharing or a conflict-aware method. The exact toy vectors and drawing are original, informed by <a href="https://proceedings.mlr.press/v80/chen18a.html">GradNorm</a> and <a href="https://arxiv.org/abs/2001.06782">PCGrad</a>.</figcaption>
</figure>

## Loss balancing

### Fixed weights

Choose weights from product importance and tune them on held-out data. Normalize losses or gradients first so the numeric scale does not set the result by accident.

Fixed weights are simple and stable. They may fail when tasks learn at different speeds.

### Uncertainty weighting

Learn a noise parameter for each task and weight noisier tasks less. In one common form, each loss is scaled by an inverse variance plus a term that prevents variance from growing without bound.

This method assumes the noise model fits the tasks. It does not encode product value.

### Gradient balancing

GradNorm adjusts task weights to balance gradient magnitudes and relative training rates. PCGrad projects away a conflicting component between task gradients.

These methods can help optimization, but they add state and tuning. Compare them with a clear fixed-weight baseline.

## Missing labels and sampling

Not every example has every task label. Use a label mask:

$$
L=\sum_t w_t\frac{\sum_i m_{it}L_{it}}{\sum_i m_{it}},
$$

where $m_{it}=1$ only when example $i$ has task $t$ label.

Do not fill a missing label with zero. That creates false negatives.

Oversampling a sparse task changes its apparent prevalence. If a head must output probabilities, calibrate it on the deployment distribution after training.

## MMoE and PLE

A multi-gate mixture of experts (MMoE) uses shared experts and a separate gate for each task. Each gate chooses a different mixture of expert outputs.

Progressive layered extraction (PLE) separates shared experts from task-specific experts across several layers. It gives each task a private path while retaining shared information.

These architectures help when some features are shared and others are task-specific. They do not remove the need for sound labels, weights, and evaluation.

## Multi-objective ranking

A recommender may predict click, watch time, completion, save, survey response, and return rate. The task heads are not the final product objective.

A serving score may combine calibrated predictions:

$$
S(x)=\sum_t \alpha_t \hat{p}_t(x).
$$

The training weights $w_t$ control representation learning. The serving weights $\alpha_t$ express product tradeoffs. Keep them separate and validate both online.

Long-term tasks are delayed and selective. Their heads need maturity checks and may train on a different eligible sample.

## Worked example

A video ranker predicts clicks and next-week return. Click labels are abundant; return labels are sparse and delayed.

After adding click prediction, click AUC rises while return calibration worsens. The shared trunk receives much larger click gradients.

First mask immature return labels and normalize loss contributions. Then compare fixed weights, balanced gradients, and a small task-specific branch. Choose the simplest change that improves return without unacceptable click loss.

## Evaluation

Report:

- every task metric and calibration curve;
- the primary product outcome;
- task gradient norms and conflict by layer;
- performance by label availability and maturity;
- serving cost and latency;
- ablations for each auxiliary task;
- online tradeoff curves.

A better average across tasks can hide a regression in the primary outcome.

## In an interview

Use this order:

1. Define each task, label source, and product role.
2. Start with a shared trunk and task heads.
3. Normalize losses and mask missing labels.
4. Measure per-task results and gradient interaction.
5. Explain fixed weights, adaptive balancing, MMoE, and PLE.
6. Separate training weights from serving weights.
7. Validate the product tradeoff online.

## Common mistakes

- Giving every raw loss weight 1 without checking scale.
- Treating missing labels as negative labels.
- Reporting only a combined metric.
- Adding task-specific architecture before measuring interference.
- Confusing training loss weights with product utility weights.
- Combining uncalibrated task-head probabilities.
- Assuming related task names imply compatible gradients.

## Practice next

Use this framework in [loss-function selection](/questions/how-to-choose-loss-function/), [YouTube recommendation design](/questions/design-youtube-recommender/), [Spotify homepage design](/questions/design-spotify-homepage/), and [calibration](/concepts/calibration/).
