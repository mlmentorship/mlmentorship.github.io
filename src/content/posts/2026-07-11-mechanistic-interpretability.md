---
title: "Mechanistic interpretability"
description: "Use causal interventions, not attractive visualizations, to identify internal computations that produce model behavior."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Mechanistic interpretability tries to identify the internal computations that cause a model behavior, using representations, components, and causal interventions rather than only input-output attribution.

SHAP or saliency can tell you which input features correlate with one prediction. Frontier-model questions are often different: where is a behavior represented, which components transform it, and does changing that internal state change the output as predicted?

The word "mechanistic" should raise the evidence bar. A neuron that activates on French text is a correlate. If patching that activation transfers a relevant behavior under controlled conditions, the causal case is stronger.

## The evidence ladder

1. **Observation:** an activation, feature, head, or layer correlates with a behavior.
2. **Localization:** the signal appears in a repeatable component and time step.
3. **Intervention:** ablation, patching, steering, or editing changes the behavior.
4. **Specificity:** the intervention affects the predicted behavior more than controls.
5. **Composition:** multiple components form a circuit whose interactions predict results.
6. **Generalization:** the mechanism survives new prompts, paraphrases, domains, and model instances.

Most impressive pictures live near steps one or two. Strong claims need the later steps.

## Core tools

### Logit lens and tuned lenses

Project intermediate residual-stream states through the unembedding to inspect token preferences. This gives a readable trajectory but does not prove that the model uses the decoded representation in the final computation.

### Activation patching

Run a clean and corrupted input, then replace an internal activation in one run with the corresponding activation from the other. If the output recovers, the patched component carries causally relevant information for that contrast.

Results depend on the corruption, metric, patch location, and distribution shift caused by the intervention.

<!-- visual:activation-patching-controlled-contrast -->
<figure class="learning-figure plot-panel" aria-labelledby="activation-patching-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="activation-patching-visual-title">Trace one same-site state transfer, then limit the causal claim to that intervention.</p>
	<svg viewBox="0 0 360 500" role="img" aria-labelledby="activation-patching-svg-title activation-patching-svg-desc">
		<title id="activation-patching-svg-title">Activation patching as a controlled three-run comparison</title>
		<desc id="activation-patching-svg-desc">In the clean baseline, a clean input produces activation a clean at site s and a high target metric. In the corrupted baseline, a matched corrupted input produces activation a corrupted at the same site and a low target metric. In the patched run, the corrupted input is used again, but activation a clean replaces a corrupted only at site s. The target metric recovers. This makes site s causally relevant for this contrast under this intervention, but does not by itself establish a complete circuit, necessity, sufficiency, or a natural model computation.</desc>
		<text class="viz-axis-label" x="10" y="20">RUN</text><text class="viz-axis-label" x="68" y="20" text-anchor="middle">INPUT</text><text class="viz-axis-label" x="180" y="20" text-anchor="middle">SAME SITE s</text><text class="viz-axis-label" x="300" y="20" text-anchor="middle">TARGET METRIC</text>
		<text class="viz-callout" x="10" y="54">1 · CLEAN BASELINE</text>
		<rect class="viz-node viz-node--input" x="10" y="66" width="96" height="56" rx="4"></rect><text class="viz-node-value" x="58" y="87">CLEAN INPUT</text><text class="viz-node-label" x="58" y="108">x<tspan baseline-shift="sub" font-size="9">clean</tspan></text>
		<path class="viz-axis" d="M106 94 H128"></path><path class="viz-arrow-forward" d="M134 94 l-9 -5 v10 Z"></path>
		<rect class="viz-node" x="134" y="66" width="92" height="56" rx="4"></rect><text class="viz-node-value" x="180" y="87">ACTIVATION</text><text class="viz-node-label" x="180" y="108">a<tspan baseline-shift="sub" font-size="9">clean</tspan></text>
		<path class="viz-axis" d="M226 94 H248"></path><path class="viz-arrow-forward" d="M254 94 l-9 -5 v10 Z"></path>
		<rect class="viz-node viz-node--output" x="254" y="66" width="96" height="56" rx="4"></rect><text class="viz-node-value" x="302" y="87">EXPECTED TARGET</text><text class="viz-node-label" x="302" y="108">high</text>
		<text class="viz-callout" x="10" y="158">2 · CORRUPTED BASELINE</text>
		<rect class="viz-node viz-node--input" x="10" y="170" width="96" height="56" rx="4"></rect><text class="viz-node-value" x="58" y="191">MATCHED INPUT</text><text class="viz-node-label" x="58" y="212">x<tspan baseline-shift="sub" font-size="9">corr</tspan></text>
		<path class="viz-axis" d="M106 198 H128"></path><path class="viz-arrow-forward" d="M134 198 l-9 -5 v10 Z"></path>
		<rect class="viz-node" x="134" y="170" width="92" height="56" rx="4"></rect><text class="viz-node-value" x="180" y="191">ACTIVATION</text><text class="viz-node-label" x="180" y="212">a<tspan baseline-shift="sub" font-size="9">corr</tspan></text>
		<path class="viz-axis" d="M226 198 H248"></path><path class="viz-arrow-forward" d="M254 198 l-9 -5 v10 Z"></path>
		<rect class="viz-node" x="254" y="170" width="96" height="56" rx="4"></rect><text class="viz-node-value" x="302" y="191">EXPECTED TARGET</text><text class="viz-node-label" x="302" y="212">low</text>
		<path d="M180 122 V278" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4"></path><path class="viz-arrow-backward" d="M180 286 l-5 -9 h10 Z"></path><rect class="viz-node viz-node--focus" x="108" y="242" width="144" height="30" rx="4"></rect><text class="viz-node-value" x="180" y="261">COPY CLEAN STATE AT s</text>
		<text class="viz-callout" x="10" y="316">3 · PATCHED RUN</text>
		<rect class="viz-node viz-node--input" x="10" y="328" width="96" height="64" rx="4"></rect><text class="viz-node-value" x="58" y="351">SAME CORRUPTED</text><text class="viz-node-label" x="58" y="374">x<tspan baseline-shift="sub" font-size="9">corr</tspan></text>
		<path class="viz-axis" d="M106 360 H128"></path><path class="viz-arrow-forward" d="M134 360 l-9 -5 v10 Z"></path>
		<rect class="viz-node viz-node--focus" x="134" y="328" width="92" height="64" rx="4"></rect><text class="viz-node-value" x="180" y="349">REPLACE ONLY s</text><text class="viz-node-label" x="180" y="370">a<tspan baseline-shift="sub" font-size="9">clean</tspan></text><text class="viz-edge-label" x="180" y="385">then continue</text>
		<path class="viz-axis" d="M226 360 H248"></path><path class="viz-arrow-forward" d="M254 360 l-9 -5 v10 Z"></path>
		<rect class="viz-node viz-node--output" x="254" y="328" width="96" height="64" rx="4"></rect><text class="viz-node-value" x="302" y="349">TARGET MOVES</text><text class="viz-node-label" x="302" y="370">toward clean</text><text class="viz-edge-label" x="302" y="385">recovery</text>
		<rect class="viz-node" x="10" y="420" width="340" height="64" rx="4"></rect><text class="viz-callout" x="180" y="442" text-anchor="middle">SUPPORTED: s carries relevant information</text><text class="viz-node-value" x="180" y="461">for this contrast, metric, and intervention</text><text class="viz-edge-label" x="180" y="477">not yet a complete circuit, necessity, or natural use</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> establish the clean and corrupted outputs first. In the third run, keep the corrupted input and replace only the chosen same-site activation with its clean value. If the predeclared target metric moves toward the clean result, that site carries causally relevant information for this contrast under this intervention. It is a stronger result than observing an activation, but controls and generalization are still needed before claiming a mechanism. Original schematic checked against <a href="https://arxiv.org/abs/2202.05262">Meng et al.'s causal tracing</a>, <a href="https://arxiv.org/abs/2309.16042">activation-patching best practices</a>, and the <a href="https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html">TransformerLens documentation</a>.</figcaption>
</figure>

### Ablation

Zero, mean-replace, resample, or otherwise remove a component. A behavior change suggests necessity under that intervention. Redundant circuits and out-of-distribution ablations complicate interpretation.

### Feature dictionaries and sparse autoencoders

Learn a sparse overcomplete set of features that reconstruct activations. This can separate superposed directions better than individual neurons. Feature labels remain hypotheses, and reconstruction quality plus intervention evidence matter.

### Attribution graphs

Approximate how features and components contribute through a computation. Graphs help generate circuit hypotheses but inherit approximation error and threshold choices.

## Designing a clean experiment

Choose a behavior with a measurable contrast. Build matched clean and corrupted prompts. Pre-register the output metric and controls. Localize candidate components, intervene, test specificity, then try to falsify the mechanism on new data.

Useful controls include random components, magnitude-matched directions, alternate corruptions, unrelated behaviors, and interventions with equal norm.

## Common confusions

- **"Attention is explanation."** Attention weights are one routing coefficient and can often change without an equivalent output change.
- **"A named feature is a discovered concept."** Human labels compress examples and can hide polysemantic behavior.
- **"Ablation proves sufficiency."** Removing a component tests a form of necessity, not whether it alone can produce the behavior.
- **"Patching is naturalistic."** Replacing internal state can move the model off its normal activation distribution.
- **"One prompt reveals a circuit."** Mechanisms need controlled sets and generalization tests.
- **"Interpretability automatically improves safety."** A method must detect or change safety-relevant behavior reliably at useful scale.

## In an interview

Separate observation, causal intervention, and generalization. State the behavior, contrast, metric, intervention, controls, and what result would falsify the proposed circuit.

*Related: [model interpretability](/concepts/model-interpretability/), [chain-of-thought monitorability](/concepts/chain-of-thought-monitorability/), and [design an ablation study](/questions/design-ablation-study/).*
