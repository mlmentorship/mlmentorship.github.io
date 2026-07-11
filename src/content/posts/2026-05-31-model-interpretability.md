---
title: "Model interpretability"
description: "How to explain a model's predictions: the split between intrinsic and post-hoc methods, global vs local, and the four techniques interviewers expect (feature importance, SHAP, LIME, and saliency / Grad-CAM)."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Interpretability is the set of methods for explaining **what a model learned** (global) or **why it made a specific prediction** (local), either by using an intrinsically transparent model or by attaching a **post-hoc** explainer to a black box.

## Why it matters

Interpretability shows up in interviews and in production for three reasons: **debugging** (is the model right for the right reasons, or exploiting a spurious feature?), **trust / regulation** (lending, healthcare, and hiring often legally require explanations), and **stakeholder buy-in**. It's also a common "you shipped a model, the PM asks *why did it reject this user*, what do you do?" scenario.

## The two axes

| | **Global** (whole model) | **Local** (one prediction) |
| --- | --- | --- |
| **Intrinsic** | Linear coefficients, tree splits, GAM shape functions | A single decision path in a tree |
| **Post-hoc** | Permutation importance, PDP / ALE | SHAP, LIME, saliency maps, counterfactuals |

- **Intrinsic vs post-hoc**: use a transparent model, or explain a black box after the fact.
- **Global vs local**: explain the model overall, or one specific decision.

## The four techniques to know

### 1. Feature importance

- **Tree split / gain importance**: how much each feature reduced impurity across splits. Cheap but **biased toward high-cardinality features** and computed on training data.
- **Permutation importance**: shuffle one feature's values and measure the drop in validation performance. Model-agnostic, uses held-out data, but **misleading under correlated features** (shuffling one of two correlated features looks unimportant because the other compensates).

### 2. LIME (Local Interpretable Model-agnostic Explanations)

Fit a simple, interpretable surrogate (usually sparse linear) to the black box **in the neighborhood of one point**: perturb the input, get the model's predictions, weight perturbations by proximity, and fit a local linear model. Output: per-feature weights for *this* prediction. Fast and intuitive, but explanations can be **unstable** (sensitive to the perturbation/kernel choice).

### 3. SHAP (SHapley Additive exPlanations)

Grounded in cooperative game theory: the **Shapley value** of a feature is its average marginal contribution to the prediction over all possible feature orderings. SHAP attributions are the unique solution satisfying **local accuracy** (attributions sum to prediction − baseline), **missingness**, and **consistency**.

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big(f(S \cup \{i\}) - f(S)\big).
$$

Exact Shapley values are exponential; **TreeSHAP** computes them efficiently for tree ensembles, and **KernelSHAP** approximates them model-agnostically (it's essentially LIME with the Shapley-consistent kernel and loss). SHAP is the de-facto standard for tabular explanations because it's both local (per-row) and aggregable into global importance.

### 4. Saliency / Grad-CAM (deep nets)

For images and other deep models, attribute the prediction to input regions:

- **Vanilla saliency**: gradient of the class score w.r.t. input pixels, $\partial y_c / \partial x$. Noisy.
- **Integrated Gradients**: integrate gradients along a path from a baseline to the input, satisfying sensitivity and implementation-invariance axioms.
- **Grad-CAM**: weight the final conv feature maps by the gradient of the class score flowing into them, giving a coarse class-discriminative heatmap. The standard CNN visualization.
- **Attention weights** are *not* reliable explanations: "attention is not explanation" is a known result; high attention ≠ high causal importance.

## What an interviewer expects you to say

1. Separate **intrinsic vs post-hoc** and **global vs local**; most candidates conflate them.
2. Know that **permutation importance breaks under correlated features**, and tree gain importance is biased and train-set-based.
3. Explain **SHAP = Shapley values**, that it's additive/consistent, and that TreeSHAP makes it tractable for trees.
4. For deep nets, name **Grad-CAM / Integrated Gradients** and flag that **raw attention weights aren't explanations**.
5. Bonus: mention **counterfactual explanations** ("change feature X by Δ to flip the decision") as the most actionable form for end users, and that the right method depends on audience (engineer debugging vs regulator vs end user).

## Common confusions

- **"Feature importance is causal."** It's associational. A feature can be important to the model and have no causal effect on the outcome.
- **"SHAP and LIME give the same thing."** Both are local, but SHAP has game-theoretic uniqueness guarantees; LIME's surrogate fit is heuristic and less stable.
- **"Attention shows what the model uses."** Not reliably. Attention can be redistributed without changing the output.
- **"Interpretable models are always worse."** On tabular data, well-tuned GBMs + SHAP, or even GAMs, are often both accurate and explainable. The accuracy-interpretability tradeoff is real but smaller than people assume on structured data.
- **"More explanation is better."** Explanations have an audience. A 40-feature SHAP plot helps an engineer and confuses a loan applicant who needs one actionable counterfactual.

---

*Related: [Random forests](/concepts/random-forests/), [Gradient boosting](/concepts/gradient-boosting/), [CNN architecture](/concepts/cnn-architecture/), [Calibration](/concepts/calibration/).*
