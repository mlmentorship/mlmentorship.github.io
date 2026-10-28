---
title: "How would you debug a model that's not learning?"
description: "Isolate loss, gradient, and parameter-update failures with a reproducible batch before escalating to optimization or task mismatch."
date: "2025-05-14"
draft: false
tags: ["questions"]
category: "questions"
---


Investigate a training run that appears not to learn, and propose checks that distinguish competing explanations.

## Problem

**Context.** A hypothetical model-training run shows flat loss or unchanging task performance, despite apparently completing training steps.
You can inspect a small batch, the objective, gradients, parameters, and evaluation outputs.

**Task.** Define the failing observation, prioritize hypotheses, and choose cheap tests with predicted results before changing the training system.
Explain how the investigation changes if loss falls but task performance does not.

**Constraints.** Distinguish objective computation, gradient flow, optimizer updates, and evaluation correctness.
Control randomness and change one causal factor per check; an observed correlation alone does not identify the defect.
The retained level headings organize examples, not hiring guarantees.

## What an L4 answer sounds like

"Check the learning rate, loader, loss, and optimizer" names possibilities without an observation that distinguishes them.
Changing all four together may improve loss while leaving the actual fault unidentified.

## What an L5 answer sounds like

Freeze a tiny batch, inspect its actual inputs and targets, and disable optional regularization or augmentation for the diagnostic.
State an attainable loss before asking the model to memorize it.
Two identical inputs with opposite binary labels have minimum mean cross-entropy $\log2$, not zero, for a deterministic shared prediction.

Use a scalar control with parameter $w=0$, loss $L=\tfrac12(w-3)^2$, and SGD rate $0.1$.
The initial loss is $4.5$, gradient $-3$, expected updated parameter $0.3$, and new loss $3.645$.
If the gradient is $-3$ but $w$ remains zero, the gradient alone has not established a working update.
Inspect optimizer membership, whether the step executed, resets, finite-value guards, and frozen parameters before changing capacity.

For the real batch, record the hand-computed loss, reduction, gradient, and before/after parameter values.
Keep a hypothesis/test/result ledger; change one supported cause, then repeat the same check.

<p class="visual-kicker">Diagnostic path</p>
<!-- visual:model-not-learning-diagnostic-path -->
<figure class="learning-figure editorial-illustration" aria-labelledby="model-not-learning-diagnostic-path-edition-title"><p class="visual-title" id="model-not-learning-diagnostic-path-edition-title">Separate a broken training path from a full-data generalization gap</p><svg viewBox="0 0 360 475" role="img" aria-labelledby="model-not-learning-diagnostic-path-edition-svg-title model-not-learning-diagnostic-path-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="model-not-learning-diagnostic-path-edition-svg-title">Separate a broken training path from a full-data generalization gap</title><desc id="model-not-learning-diagnostic-path-edition-svg-desc">Three illustrative loss plots share a zero-to-two vertical scale. One tiny fixed batch reaches an attainable minimum; another stays stuck and requires training-path checks. A full-data example improves training while validation diverges. The gap is diagnostic evidence, not proof of a particular cause.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative loss traces; same vertical</tspan><tspan x="180" dy="19">scale</tspan></text><text x="18" y="58" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Tiny fixed batch: reaches attainable minimum</tspan></text><path d="M45 78L45 149" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M45 149L325 149" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="32" y="85" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="32" dy="0">2</tspan></text><text x="32" y="153" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="32" dy="0">0</tspan></text><polyline points="45,83 115,102.80000000000001 185,129.2 255,142.4 325,149" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></polyline><text x="18" y="184" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Tiny batch stuck: inspect training</tspan></text><path d="M45 204L45 275" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M45 275L325 275" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="32" y="211" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="32" dy="0">2</tspan></text><text x="32" y="279" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="32" dy="0">0</tspan></text><polyline points="45,209 115,209 185,210.65 255,209 325,209" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></polyline><text x="18" y="310" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Full data: growing gap is a clue, not a cause</tspan></text><path d="M45 330L45 401" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M45 401L325 401" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="32" y="337" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="32" dy="0">2</tspan></text><text x="32" y="405" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="32" dy="0">0</tspan></text><polyline points="45,335 115,354.8 185,374.6 255,387.8 325,394.4" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></polyline><polyline points="45,335 115,351.5 185,354.8 255,351.5 325,348.2" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-dasharray:5 3"></polyline><text x="180" y="456" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">solid: training dashed: validation</tspan></text></svg><figcaption><strong>Read it this way:</strong> First check whether the tiny batch has an attainable minimum. A widening full-data gap suggests hypotheses, not a unique cause. The curves are illustrative. <span class="visual-reader-check" data-reader-check="model-not-learning-diagnostic-path"><strong>Check:</strong> If the tiny batch cannot reach its justified minimum, which investigation comes before generalization tuning?</span></figcaption></figure>

## What an L6 answer sounds like

Extend the evidence, not the number of simultaneous interventions.
Check target conventions, masks, sum-versus-mean reduction, accumulation counts, train/eval mode, and finite gradients alongside parameter changes.
Evaluation mode does not universally disable gradients; its effect depends on the layers.
Measure device throughput and completed optimizer steps separately from mathematical learning progress.

### The Loss Improves but the Task Does Not

**New task:** Diagnose useful-output failure after verifying the update path.

**Picture and change:** Construct low-loss failure cases and inspect targets, masks, scoring, and deployment objectives.

**Check:** Ninety-nine padding tokens improving from loss one to zero can hide one important token remaining at loss one.
The all-token mean falls from one to $0.01$ without fixing that important prediction.

**Use when:** The optimized measurement and required task disagree; this does not establish a gradient fault.

### Training Improves but Validation Degrades

**New task:** Explain a growing gap without assuming one cause.

**Picture and change:** Audit split validity, target consistency, distribution differences, and evaluation mode before testing ordinary overfitting.

**Check:** A validation-only class cannot be supplied merely by adding dropout.

**Use when:** The two datasets may no longer support the same inference.
Regularize or change data only after a discriminating check supports that intervention.

<!-- learning:core-end -->
<div data-learning-extension>

## The tells that get you a strong-hire vote

Use the retained heading as a response-quality checklist, not a prediction of an interviewer vote.
Show an ordered test, its possible outcomes, actual observed evidence, and the next conditional action.
Discuss real experience honestly; do not invent debugging history to imitate these examples.

## The tells that get you down-leveled

An unordered optimizer/LR/initialization/loader sweep cannot isolate a fault, even if the final run improves.
Likewise, frozen predictions can reflect update failures, saturation, insensitive measurements, or logging errors rather than one proven cause.

## A follow-up the interviewer often asks

For slow progress, compare loss per completed update and wall-clock step rate before selecting an optimization or systems hypothesis.
Inspect gradient scale, numerical precision, conditioning, data coverage, and update magnitude where evidence points.
Squared error on class probabilities can be a legitimate Brier objective; classification alone does not make MSE invalid.
It can still differ from cross-entropy in gradients, calibration behavior, and the intended decision target.

---

*Related: [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), [Walk me through bias-variance tradeoff](/questions/bias-variance-tradeoff/).*

</div>
