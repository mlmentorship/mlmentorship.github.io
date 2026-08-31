---
title: "Decision trees"
description: "Recursively split the feature space along axis-aligned thresholds chosen to maximize a purity criterion. The base learner of GBDT and random forests."
date: "2026-04-19"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **decision tree** partitions the feature space by a sequence of axis-aligned threshold tests on individual features, assigning a constant prediction (class probability for classification, mean target for regression) to each leaf. Trained greedily by choosing splits that maximize information gain or equivalently minimize a purity / impurity criterion.

Single trees are rarely the production model. Variance is too high. But they are the building block of the dominant tabular learners: **gradient boosting** (xgboost, lightgbm, catboost) and **random forests**. Understanding tree training is the prerequisite for using or tuning either.

Trees naturally handle missing values, mixed-type features, non-linear interactions, and require essentially no feature scaling. This is why they remain the strongest baseline on heterogeneous tabular data.

## The algorithm (CART, [Breiman et al., 1984](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418))

For each node:

1. For each feature $j$ and each candidate threshold $t$:
   - Split data into $\{x : x_j \le t\}$ (left child) and $\{x : x_j > t\}$ (right child).
   - Compute the impurity of each child (Gini, entropy, or MSE).
   - Compute the weighted impurity reduction.
2. Pick the $(j, t)$ with maximum reduction.
3. Recurse on each child until a stopping criterion (max depth, min samples per leaf, no further reduction).

The greedy choice is locally optimal but not globally; finding the globally optimal tree is NP-hard.

**Learning objective:** map each root-to-leaf rule path to one axis-aligned region of feature space and its constant leaf prediction.

<!-- visual:decision-tree-partitions-feature-space -->
<figure class="learning-figure plot-panel" aria-labelledby="decision-tree-partition-title">
	<p class="visual-kicker">Rules become regions</p>
	<p class="visual-title" id="decision-tree-partition-title">A child split cuts only the region that reaches it.</p>
	<svg viewBox="0 0 360 440" role="img" aria-labelledby="decision-tree-svg-title decision-tree-svg-desc">
		<title id="decision-tree-svg-title">A decision tree and its matching feature-space partition</title>
		<desc id="decision-tree-svg-desc">The upper plot contains four circle samples and four triangle samples. A full-height vertical root split at feature one equal to four sends four points left and four right. Only the right region receives a horizontal child split at feature two equal to three. The matching tree below has a root test feature one less than or equal to four. Its yes branch ends at a leaf with triangle probability one quarter. Its no branch tests feature two less than or equal to three, producing a pure triangle leaf below and a pure circle leaf above. A highlighted new point follows the no then yes path to the triangle leaf.</desc>
		<rect class="viz-plot-bg" x="42" y="30" width="286" height="184" rx="4"></rect>
		<path class="viz-gridline" d="M42 76H328M42 122H328M42 168H328M99 30V214M156 30V214M213 30V214M270 30V214"></path>
		<path class="viz-axis" d="M42 24V214H334"></path>
		<text class="viz-label" x="185" y="236" text-anchor="middle">feature 1</text>
		<text class="viz-label" x="13" y="125" text-anchor="middle" transform="rotate(-90 13 125)">feature 2</text>
		<path d="M185 30V214" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3"></path>
		<text class="viz-callout" x="179" y="45" text-anchor="end">root: x1 ≤ 4</text>
		<path d="M185 122H328" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-dasharray:6 4"></path>
		<text class="viz-callout" x="321" y="116" text-anchor="end">child: x2 ≤ 3</text>
		<g style="fill:var(--viz-surface);stroke:var(--viz-edge);stroke-width:2">
			<circle cx="76" cy="64" r="6"></circle><circle cx="113" cy="99" r="6"></circle><circle cx="145" cy="55" r="6"></circle><circle cx="288" cy="70" r="6"></circle>
			<path d="M92 143L99 156H85Z"></path><path d="M224 152L231 165H217Z"></path><path d="M252 177L259 190H245Z"></path><path d="M278 174L285 187H271Z"></path>
		</g>
		<path d="M249 183L257 197H241Z" style="fill:var(--viz-focus-fill);stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
		<text class="viz-callout" x="258" y="204">new point</text>
		<text class="viz-axis-label" x="42" y="19">8 TRAINING POINTS: ○ CLASS 0, △ CLASS 1</text>
		<text class="viz-axis-label" x="180" y="264" text-anchor="middle">THE SAME PARTITION, WRITTEN AS RULES</text>
		<path d="M180 301L91 337M180 301L269 337M269 371L222 405M269 371L316 405" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
		<rect class="viz-node viz-node--focus" x="120" y="276" width="120" height="32" rx="4"></rect>
		<text class="viz-callout" x="180" y="297" text-anchor="middle">x1 ≤ 4?</text>
		<text class="viz-label" x="117" y="325" text-anchor="end">yes</text>
		<text class="viz-label" x="243" y="325">no</text>
		<rect class="viz-node" x="35" y="337" width="112" height="36" rx="4"></rect>
		<text class="viz-callout" x="91" y="352" text-anchor="middle">leaf L</text>
		<text class="viz-label" x="91" y="367" text-anchor="middle">P(△) = 1/4</text>
		<rect class="viz-node viz-node--input" x="209" y="337" width="120" height="36" rx="4"></rect>
		<text class="viz-callout" x="269" y="359" text-anchor="middle">x2 ≤ 3?</text>
		<text class="viz-label" x="223" y="391" text-anchor="middle">yes</text>
		<text class="viz-label" x="315" y="391" text-anchor="middle">no</text>
		<rect class="viz-node viz-node--output" x="171" y="405" width="102" height="30" rx="4"></rect>
		<text class="viz-callout" x="222" y="425" text-anchor="middle">leaf R1: P(△)=1</text>
		<rect class="viz-node" x="278" y="405" width="76" height="30" rx="4"></rect>
		<text class="viz-callout" x="316" y="425" text-anchor="middle">R2: P(△)=0</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> begin with the solid vertical root split: every point goes left or right. The dashed horizontal test belongs only to the right child, so it cuts only that half-plane. A root-to-leaf path is therefore an intersection of threshold rules, forming a rectangle here, and every point in that rectangle receives the leaf's constant class probability. The highlighted triangle follows “no, then yes” to leaf R1. Original schematic checked against the <a href="https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation">scikit-learn tree formulation</a> and <a href="https://hastie.su.domains/ElemStatLearn/">The Elements of Statistical Learning</a>.</figcaption>
</figure>

## Split criteria

For classification at a node with class proportions $p_k$:

- **Gini impurity**: $1 - \sum_k p_k^2$. Equals expected error of random labeling proportional to $p$.
- **Entropy**: $-\sum_k p_k \log p_k$. Information gain = parent entropy minus weighted child entropies.

Gini and entropy give nearly identical splits in practice; Gini is slightly cheaper.

For regression: variance of the target within the node, equivalent to MSE under constant prediction.

## Stopping and pruning

Stop splitting when:

- Max depth reached.
- Number of samples at the node falls below a threshold.
- Best impurity reduction is below a threshold.
- All samples have the same target.

**Pruning**: train deep, then collapse branches that don't reduce held-out error. Cost-complexity pruning (CCP) trades depth against penalty $\alpha \times |\text{leaves}|$.

In practice with ensembles (boosting, forests), individual trees are kept shallow (depth 6–8 for boosting, depth limited or full for forests with bagging variance averaging).

## Strengths and weaknesses

**Strengths:**

- Handles non-linear interactions for free.
- No feature scaling needed.
- Handles missing values (with surrogate splits or built-in handling in xgboost/lightgbm).
- Handles mixed numeric / categorical (with appropriate encoding).
- Interpretable: every prediction is a path of explicit rules.

**Weaknesses:**

- High variance: small data perturbations change splits dramatically.
- Greedy: locally optimal, not globally.
- Axis-aligned splits make it hard to capture rotated decision boundaries (need oblique trees or feature engineering).
- Single trees are weak. Usually combined into ensembles.

## Categorical features

Two main approaches:

- **One-hot**: each level becomes a binary feature. Slow; trees grow many useless branches.
- **Native handling** (lightgbm, catboost): pre-sort levels by target mean, then split as if the feature were numeric. Much faster and often more accurate.

## Common pitfalls

- **Letting trees grow unbounded on small data.** Memorizes training set; high variance.
- **Treating categorical features as numeric without one-hot or native handling.** Tree splits are ordered; "encoding" a categorical with arbitrary integer codes imposes a meaningless order.
- **Comparing single trees against ensembles unfairly.** Single trees should not beat boosted forests; check you are evaluating the right model class.
- **Reading feature importance from a single split.** Importance from a single tree is noisy; use ensemble averages or permutation importance.

## Related

- [Random forests](/concepts/random-forests/). Bagged ensembles of trees.
- [Gradient boosting](/concepts/gradient-boosting/). Boosted ensembles of trees.
