---
title: "Decision trees"
description: "Compute weighted split improvement, trace a root-to-leaf region, and test larger leaves, regression targets, and a greedy XOR failure."
date: "2026-04-19"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A decision tree predicts a constant value within each terminal feature region.
Training chooses threshold splits greedily, using a stated criterion and stopping rules.

**Task:** the diagram has four circles, class zero, and four triangles, class one.
Candidate root $x_1\le4$ puts three circles and one triangle left; one circle and three triangles go right.
Calculate its Gini reduction and predict a new point $(6,2)$ after the right-only child split $x_2\le3$.

The drawn split is a candidate, not a proven best threshold among unenumerated alternatives.

## The algorithm (CART, [Breiman et al., 1984](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418))

1. Enumerate allowed feature thresholds, usually midpoints between distinct ordered numeric values.
2. Route $x_j\le t$ left and $x_j>t$ right; reject candidates violating leaf constraints.
3. Choose maximum sample-weighted impurity reduction, then recurse under the stopping rules.

This local choice need not produce the globally best tree.

**Learning objective:** map each root-to-leaf rule path to one axis-aligned region of feature space and its constant leaf prediction.

<!-- visual:decision-tree-partitions-feature-space -->
<figure class="learning-figure plot-panel" aria-labelledby="decision-tree-partition-title">
	<p class="visual-kicker">Rules become regions</p>
	<p class="visual-title" id="decision-tree-partition-title">The second split affects only the root's right-hand region</p><div class="visual-argument" data-visual-argument="decision-tree-partitions-feature-space" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="The second split affects only one root branch. The root sends every x&lt;=0 point to A. Splitting y on the right branch changes only the x&gt;0 region."><title>The second split affects only one root branch</title><desc>The root sends every x&lt;=0 point to A. Splitting y on the right branch changes only the x&gt;0 region.</desc><text x="128.62" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="128.62" dy="0">x&lt;=0</tspan></text><text x="269.54" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="269.54" dy="0">x&gt;0</tspan></text><text x="46.16" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">y&gt;1</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="58.16" y="36" width="135.92000000000002" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="126.12" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="126.12" dy="0">A</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="199.08" y="36" width="135.92000000000002" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="267.04" y="71" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="267.04" dy="0">C</tspan></text><path d="M204.08 89L330 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="46.16" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">y&lt;=1</tspan></text><g data-proof-status="value"><rect data-proof-cell="1,0" x="58.16" y="100" width="135.92000000000002" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="126.12" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="126.12" dy="0">A</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="199.08" y="100" width="135.92000000000002" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="267.04" y="135" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="267.04" dy="0">B</tspan></text><path d="M204.08 153L330 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
		<g style="fill:var(--viz-neutral-bg);stroke:var(--viz-edge);stroke-width:2">
			<circle cx="76" cy="64" r="6"></circle><circle cx="113" cy="99" r="6"></circle><circle cx="145" cy="55" r="6"></circle><circle cx="288" cy="70" r="6"></circle>
			<path d="M92 143L99 156H85Z"></path><path d="M224 152L231 165H217Z"></path><path d="M252 177L259 190H245Z"></path><path d="M278 174L285 187H271Z"></path>
		</g>
		<path d="M249 183L257 197H241Z" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
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
	<div class="visual-derivation-caption">the solid root split routes every point. The dashed child test cuts only the right region. Intersecting the path's predicates gives its leaf region and constant empirical class fraction. The highlighted point follows “no, then yes” to R1. Sources: <a href="https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation">scikit-learn's formulation</a> and <a href="https://hastie.su.domains/ElemStatLearn/">The Elements of Statistical Learning</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="decision-tree-partitions-feature-space"><strong>Check:</strong> Why is the second split applied only on the right side of the root boundary?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="decision-tree-partitions-feature-space">The root sends every x&lt;=0 point to A. Splitting y on the right branch changes only the x&gt;0 region.</span></figcaption>
</figure>

## Split criteria

For class proportions $p_k$, Gini impurity is $I=1-\sum_kp_k^2$.
Use

$$
\Delta I=I(parent)-\frac{n_L}{n}I(L)-\frac{n_R}{n}I(R).
$$

| State | Calculation | Gini |
| --- | --- | --- |
| Parent | $1-(1/2)^2-(1/2)^2$ | $0.5$ |
| Each root child | $1-(3/4)^2-(1/4)^2$ | $0.375$ |
| Weighted root children | $(4/8)(0.375)+(4/8)(0.375)$ | $0.375$ |

Root improvement is $0.5-0.375=0.125$.
The right child separates three triangles from one circle, leaving total weighted impurity $(4/8)(0.375)=0.1875$.
Point $(6,2)$ reaches that pure-triangle R1 leaf, with empirical triangle fraction $1$.

Sample weights matter: a separate pure child of size three and impurity-$0.32$ child of size five give weighted impurity $0.2$.
An unweighted mean gives $0.16$, incorrectly treating unequal child sizes equally.

Entropy $-\sum_kp_k\log p_k$ gives entropy information gain, not the same numerical quantity as Gini reduction.
For regression, a leaf mean minimizes its summed squared error.

## Stopping and pruning

**New task:** require at least four samples per leaf.
The 4/4 root remains legal; its proposed 3/1 right split does not.
**Check:** $(6,2)$ now receives the unsplit right leaf's triangle fraction $3/4$, not $1$.
**Use when:** tiny leaves provide insufficient support or excessive variance.

**New task:** predict numeric targets with $x_1=[1,2,5,6]$, $y=[1,1,3,3]$, and threshold $4$.
Replace class fractions and Gini with means and squared residuals.
**Check:** the unsplit mean $2$ has SSE $4$; leaf means $1$ and $3$ give SSE zero.
**Use when:** the output is a continuous response.

Depth, minimum leaf size, purity, and improvement limits stop growth.
Cost-complexity pruning compares subtrees using $R(T)+\alpha|leaves(T)|$; select $\alpha$ using validation, not test outcomes.

**Failure:** either first axis split on four XOR corners has zero gain, although a depth-two tree separates them.
Stopping on nonpositive immediate gain can miss that interaction.

**Reconstruct:** at $x_1=4$ exactly, does the right child inspect $x_2$?

<details><summary>Answer</summary>

No. The $\le$ rule routes left; a child test applies only to the region reaching it.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Strengths and weaknesses

Paths provide explicit rules and can represent interactions through successive splits.
Strictly increasing numeric transformations preserve candidate partitions when ordering and implementation arithmetic are preserved.
Rotated boundaries may require many axis-aligned regions; feature engineering or oblique tests changes that representation.
Small data changes can change selected splits.
Neither a single tree nor an ensemble has a universal production ranking.

## Categorical features

One-hot encoding and native categorical splitting define different candidate partitions and costs.
Arbitrary integer codes impose an ordering that may not match category meaning.
Target-based category statistics require leakage controls.
Library algorithms, including CatBoost's ordered statistics, are not interchangeable with simply sorting all categories by target mean.

## Common pitfalls

Missing-value routing needs an explicit supported strategy: imputation, learned default directions, or surrogate splits, depending on implementation.
Do not assume every tree accepts missing or categorical values.
Empirical leaf fractions are not guaranteed calibrated probabilities for new populations.
Feature importance can reflect split-selection biases; evaluate stability and held-out perturbations before interpreting it.

## Related

- [Random forests](/concepts/random-forests/). Bagged ensembles of trees.
- [Gradient boosting](/concepts/gradient-boosting/). Boosted ensembles of trees.

</div>
