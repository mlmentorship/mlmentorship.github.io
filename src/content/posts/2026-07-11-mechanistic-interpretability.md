---
title: "Mechanistic interpretability"
description: "Use causal interventions, not attractive visualizations, to identify internal computations that produce model behavior."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Mechanistic interpretability tries to identify the internal computations that cause a model behavior, using representations, components, and causal interventions rather than only input-output attribution.

## Why it matters

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
