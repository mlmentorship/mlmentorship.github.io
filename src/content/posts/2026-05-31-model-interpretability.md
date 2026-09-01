---
title: "Model interpretability"
description: "How to explain a model's predictions: the split between intrinsic and post-hoc methods, global vs local, and the four techniques interviewers expect (feature importance, SHAP, LIME, and saliency / Grad-CAM)."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Interpretability is the set of methods for explaining **what a model learned** (global) or **why it made a specific prediction** (local), either by using an intrinsically transparent model or by attaching a **post-hoc** explainer to a black box.

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

<!-- visual:shap-all-orderings-credit -->
<figure class="learning-figure plot-panel" aria-labelledby="shap-orderings-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="shap-orderings-visual-title">See why one “with versus without” comparison is not yet a SHAP value.</p>
	<svg viewBox="0 0 360 500" role="img" aria-labelledby="shap-orderings-svg-title shap-orderings-svg-desc">
		<title id="shap-orderings-svg-title">Feature B's Shapley value across every ordering of three features</title>
		<desc id="shap-orderings-svg-desc">For a toy model with features A, B, and C, all six feature orderings are listed. Feature B is outlined and labelled in every row. Its marginal contribution is measured when it joins the features before it: after A; after A and C; before both features in two orderings; after C and A; and after C. Feature B's Shapley value is the average of those six context-dependent differences. Repeating this calculation for A and C produces attributions whose sum equals the prediction minus the baseline.</desc>
		<text class="viz-axis-label" x="12" y="20">ORDER</text><text class="viz-axis-label" x="348" y="20" text-anchor="end">B JOINS AFTER S → CONTRIBUTION</text>
		<rect class="viz-plot-bg" x="8" y="32" width="344" height="336" rx="4"></rect>
		<g transform="translate(0 0)">
			<text class="viz-callout" x="18" y="62">1</text><rect class="viz-node" x="42" y="42" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="61" y="62">A</text><rect class="viz-node viz-node--focus" x="86" y="42" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="105" y="62">B</text><rect class="viz-node" x="130" y="42" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="149" y="62">C</text><text class="viz-label" x="184" y="61">S={A}: f(A,B) − f(A)</text>
			<text class="viz-callout" x="18" y="116">2</text><rect class="viz-node" x="42" y="96" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="61" y="116">A</text><rect class="viz-node" x="86" y="96" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="105" y="116">C</text><rect class="viz-node viz-node--focus" x="130" y="96" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="149" y="116">B</text><text class="viz-label" x="184" y="115">S={A,C}: f(A,B,C) − f(A,C)</text>
			<text class="viz-callout" x="18" y="170">3</text><rect class="viz-node viz-node--focus" x="42" y="150" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="61" y="170">B</text><rect class="viz-node" x="86" y="150" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="105" y="170">A</text><rect class="viz-node" x="130" y="150" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="149" y="170">C</text><text class="viz-label" x="184" y="169">S=∅: f(B) − f(∅)</text>
			<text class="viz-callout" x="18" y="224">4</text><rect class="viz-node viz-node--focus" x="42" y="204" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="61" y="224">B</text><rect class="viz-node" x="86" y="204" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="105" y="224">C</text><rect class="viz-node" x="130" y="204" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="149" y="224">A</text><text class="viz-label" x="184" y="223">S=∅: f(B) − f(∅)</text>
			<text class="viz-callout" x="18" y="278">5</text><rect class="viz-node" x="42" y="258" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="61" y="278">C</text><rect class="viz-node" x="86" y="258" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="105" y="278">A</text><rect class="viz-node viz-node--focus" x="130" y="258" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="149" y="278">B</text><text class="viz-label" x="184" y="277">S={A,C}: f(A,B,C) − f(A,C)</text>
			<text class="viz-callout" x="18" y="332">6</text><rect class="viz-node" x="42" y="312" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="61" y="332">C</text><rect class="viz-node viz-node--focus" x="86" y="312" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="105" y="332">B</text><rect class="viz-node" x="130" y="312" width="38" height="30" rx="4"></rect><text class="viz-node-value" x="149" y="332">A</text><text class="viz-label" x="184" y="331">S={C}: f(B,C) − f(C)</text>
		</g>
		<path d="M180 378V396" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path><path class="viz-arrow-forward" d="M180 404 l-5 -9 h10 Z"></path>
		<rect class="viz-node viz-node--focus" x="18" y="406" width="324" height="46" rx="4"></rect><text class="viz-callout" x="180" y="426" text-anchor="middle">φ<tspan baseline-shift="sub" font-size="9">B</tspan> = average of all six B contributions</text><text class="viz-label" x="180" y="443" text-anchor="middle">every possible predecessor context gets its fair weight</text>
		<rect class="viz-node" x="18" y="462" width="324" height="30" rx="4"></rect><text class="viz-node-value" x="180" y="481" text-anchor="middle">φ<tspan baseline-shift="sub" font-size="9">A</tspan> + φ<tspan baseline-shift="sub" font-size="9">B</tspan> + φ<tspan baseline-shift="sub" font-size="9">C</tspan> = f(A,B,C) − f(∅)</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> locate B in each of the six possible orders, then compare the model immediately before and after B joins. Because B can add different value after different feature sets, a single removal test is only one context. Average all six marginal contributions to obtain φ<sub>B</sub>; repeat for A and C, and the three attributions sum to the prediction minus its baseline. This is an original schematic checked against <a href="https://arxiv.org/abs/1705.07874">Lundberg and Lee's SHAP formulation</a> and the <a href="https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html">official SHAP tutorial</a>.</figcaption>
</figure>

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
