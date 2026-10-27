---
title: "Regularization: diagnose the gap and choose a constraint"
description: "Check train/validation comparability, distinguish sparse and small coefficients, and return the best validated state instead of stacking penalties."
date: "2026-01-30"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Regularization changes fitting to improve generalization under stated assumptions.
A validation error alone does not identify overfitting.
Compare unregularized training and validation losses in evaluation mode, with matched metrics and preprocessing.
Check leakage, split shift, label conventions, and optimization before selecting a constraint.

**Task:** fit independent coordinates with $L_{data}(\theta)=\frac12\sum_i(\theta_i-a_i)^2$, where $a=[0.1,2]$.
Choose between small coefficients and exact zeros at penalty strength $\lambda=0.2$.
These are analytic fixtures, not measured generalization gains.

<!-- visual:regularization-diagnose-and-target -->
<figure class="learning-figure" aria-labelledby="regularization-choice-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="regularization-choice-title">Does the evidence call for regularization, and which lever targets the failure?</p>
	<div class="visual-grid--two" role="group" aria-label="A diagnosis panel compares training and validation behavior before a second panel matches an observed overfitting mechanism to one regularization lever">
		<section class="visual-panel">
			<h4>1 - DIAGNOSE BEFORE ADDING</h4>
			<p><strong>Train poor + validation poor</strong><br />Check optimization, representation, data quality, and objective alignment before tightening constraints.</p>
			<p><strong>Train good + validation worse</strong><br />A gap warrants split and mode checks before calling it overfitting. Test the proposed constraint on held-out data.</p>
			<p><strong>Train good + validation good</strong><br />Do not add a penalty by default. More regularization can create underfitting.</p>
		</section>
		<section class="visual-panel">
			<h4>2 - TARGET ONE SOURCE OF CAPACITY</h4>
			<p><strong>Missing known invariance &rarr; augmentation</strong><br />Teach label-preserving input variation.</p>
			<p><strong>Magnitude constraint &rarr; test decoupled decay</strong><br />Shrink selected parameters directly; distinguish AdamW from a loss penalty.</p>
			<p><strong>Co-adapted activations &rarr; dropout</strong><br />Randomly mask training-time activations.</p>
			<p><strong>Validation degrades with more steps &rarr; early stopping</strong><br />Restore the best held-out checkpoint.</p>
			<p><strong>Exact feature sparsity required &rarr; L1</strong><br />Use a nonsmooth penalty that can produce zeros.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare matched training and validation evidence first. A gap prompts checks; it does not prove a capacity problem. Choose one lever whose intervention tests the proposed cause, then measure held-out behavior. Optimizer, batch size, and architecture can also change fitting. Sources: <a href="https://www.deeplearningbook.org/contents/regularization.html">Goodfellow et al., chapter 7</a>, <a href="https://www.jmlr.org/papers/v15/srivastava14a.html">Srivastava et al. (2014)</a>, and <a href="https://arxiv.org/abs/1711.05101">Loshchilov and Hutter (2019)</a>.</figcaption>
</figure>

## The classical lineup

For this unit-curvature quadratic, the penalty determines the optimum:

| Added penalty | Optimum | At $\lambda=0.2$ |
| --- | --- | --- |
| $\lambda\lVert\theta\rVert_1$ | $\operatorname{sign}(a_i)\max(\lvert a_i\rvert-\lambda,0)$ | $[0,1.8]$ |
| $\lambda\lVert\theta\rVert_2^2/2$ | $a_i/(1+\lambda)$ | $[1/12,5/3]$ |

The first L1 coordinate is zero because $|a_1|\le\lambda$ satisfies its subgradient condition.
For L2, differentiating gives $(1+\lambda)\theta_i=a_i$, retaining both nonzero coordinates.
Exact L1 zeros require an objective optimum or an appropriate nonsmooth solver, not merely a few ordinary gradient steps.

## When to use what

**New task:** retain the weak feature by reducing L1 strength to $0.05$.
Move the threshold inward without changing the data quadratic.
**Check:** $[\max(0.1-0.05,0),2-0.05]=[0.05,1.95]$; neither coordinate is zero.
**Use when:** held-out evidence and the desired feature count favor retaining it.

**New task:** select a checkpoint from epoch losses $[0.8,0.6,0.65,0.7]$ on validation.
Save epoch two; patience controls stopping time, not returned weights.
**Check:** restore loss $0.6$, not the final epoch's $0.7$.
**Use when:** further fitting worsens a trustworthy held-out objective; never select on the test set.

**Failure:** flipped validation label conventions can create a gap that weight decay cannot repair.

**Reconstruct:** combine L1 strength $0.2$ with L2 strength $0.5$ in this fixture. What is the optimum?

<details><summary>Answer</summary>

Threshold to $[0,1.8]$, then divide by $1.5$, giving $[0,1.2]$.
This Elastic Net formula depends on the separable unit-curvature quadratic.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

### L2 regularization (weight decay)

Use $\lambda\|\theta\|_2^2/2$ for a loss penalty.
Its Gaussian MAP-prior interpretation depends on likelihood scaling.
[Decoupled decay](/concepts/weight-decay-vs-l2/) is not generally the same optimizer update.

### L1 regularization

The absolute-value penalty corresponds to independent Laplace priors under matching MAP assumptions.
Feature scaling and solver choice affect sparse solutions.

### L1 + L2 (Elastic Net)

Combining penalties changes both threshold and shrinkage, as the reconstruction example shows.

### Dropout

Follow the [mask-expectation calculation](/concepts/dropout/) rather than treating nonlinear inference as exact ensemble averaging.
Placement and probability are architecture-specific.
Monte Carlo dropout deliberately retains sampling for a separate uncertainty procedure.

### Early stopping

Finite optimization time changes the learned solution.
Equivalence to an L2 penalty holds only under particular model, optimization, and spectral assumptions.

### Data augmentation

Specify an assumed label-preserving transformation; flipping an image or replacing a word can instead change the target.
Augmented samples are not additional independent observations.

## The modern view: implicit regularization

Optimizer, initialization, sampling, and architecture can bias solutions without an explicit penalty.
Small-batch or flat-minimum generalization advantages are conditional findings, not universal guarantees or optimizer rankings.

## Common confusions

More regularization can worsen fitting and generalization.
Joint changes obscure which hypothesis the experiment tested.

## Why interviewers ask

Explain the evidence, selected constraint, validation decision, and a counterexample to the diagnosis.

---

*Related: [Why does dropout work?](/questions/why-does-dropout-work/), [Adam, AdamW, and modern optimizer choices](/concepts/adam-and-adamw/).*

</div>
