---
title: "When would you not use cross-validation?"
description: "Cross-validation is a tool, not a default. The senior answer names the cases where it's wrong, expensive, or misleading."
date: "2025-05-25"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at every level.*

Decide when ordinary cross-validation would answer the wrong deployment question and propose a more appropriate evaluation.

## Problem

**Context.** A model may face future observations, repeated observations from the same entity, distribution shift, or expensive repeated training.
You need an estimate relevant to deployment, not merely a score from conveniently available rows.

**Task.** Explain when ordinary random folds are unsuitable and select a replacement for each failure mode.
Address evaluation population, temporal or group boundaries, training cost, and model-selection bias.

**Constraints.** Keep preprocessing and tuning inside the appropriate training boundary.
State what the chosen split estimates; neither more folds nor a structure-aware split automatically repairs deployment distribution shift.

<!-- visual:cross-validation-fit-the-question -->
<figure class="learning-figure" aria-labelledby="cross-validation-question-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="cross-validation-question-title">Choose an evaluation that matches the deployment question before choosing folds.</p>
	<div class="visual-panel" role="group" aria-label="Four ordered checks for choosing an evaluation. First check deployment representation, then dependency structure, then repeated-fit cost, and finally whether scores select the model. Each failed assumption names its replacement evaluation.">
		<ol>
			<li><strong>Does labelled data represent deployment?</strong><br>If no, use a production sample, expert set, or online experiment. Cross-validation cannot repair distribution shift.</li>
			<li><strong>Would random folds preserve dependencies?</strong><br>If no, use walk-forward splits, group folds, or spatial blocks so future, entities, and nearby locations stay separated.</li>
			<li><strong>Can you afford repeated model fits?</strong><br>If no, use one representative holdout. If yes and rows are exchangeable, ordinary cross-validation is appropriate.</li>
			<li><strong>Will these scores choose the model?</strong><br>If yes, tune inside the split and reserve an outer fold or final test. Selection must not grade itself.</li>
		</ol>
	</div>
	<figcaption><strong>Read it this way:</strong> move from population to boundary to cost to selection. Stop at the first assumption that fails and use the named replacement; structure-aware folds solve dependency leakage, not deployment shift. Original synthesis informed by <a href="https://scikit-learn.org/stable/modules/cross_validation.html">scikit-learn's cross-validation guide</a> and <a href="https://jmlr.org/papers/v11/cawley10a.html">Cawley and Talbot's analysis of selection bias</a>.</figcaption>
</figure>

## What an L4 answer sounds like

> "Cross-validation is the standard way to evaluate a model. I'd always use 5-fold or 10-fold CV unless the dataset is really large."

This is fine for textbook ML. It misses several real-world cases where CV is misleading or wasteful.

## What an L5 answer sounds like

> "I'd not use cross-validation when:
>
> 1. **The data has structure CV breaks.** Time series (random folds leak future info), grouped data (random folds leak across groups, e.g., the same patient in train and test), spatial data (nearby points are correlated). For these I'd use time-aware splits, group splits, or spatial blocking.
>
> 2. **The training cost is too high.** For deep networks with day-long training, 10x the cost is impractical. A single hold-out validation set with a representative distribution is the right call.
>
> 3. **A different evaluation answers the deployment question.** A production sample, randomized experiment, or expert-labeled set may target a more relevant population or outcome. Check selection, maturation, and assignment assumptions; none automatically supersedes every CV estimate.
>
> 4. **One split has adequate precision for the decision.** There is no universal row-count cutoff. Dependence, rare outcomes, slice coverage, candidate differences, and training cost determine whether repeated fitting adds useful information."

This is L5. You've named the cases concretely and given the right alternative for each.

## What an L6 answer sounds like

> "...a few subtler points:
>
> **CV doesn't fix distribution shift.** If your test data has a different distribution than your training data (which it usually does in production), CV on training data tells you about training-distribution generalization, not deployment performance.
>
> **Model selection reuses evidence.** Searching many candidates can select a lucky CV score. Repeated folds do not create an untouched confirmation set. Compare candidates on matched splits and use uncertainty methods that respect shared training data and entity/time dependence.
>
> **Separate selection from final evaluation.** CV can select hyperparameters, but the selected CV score is optimistic as an independent performance estimate. Nested CV evaluates the selection procedure in outer folds; a genuinely untouched, relevant test set is another option. Keep preprocessing and tuning inside each training boundary."

## Tells that get you a strong-hire vote

- You name **specific structures** that break CV (time, grouping, spatial).
- You discuss **cost** as a legitimate reason to skip.
- You mention **distribution shift** as a confound CV doesn't address.
- You bring up **nested CV** (or the alternative: separate test set) for honest hyperparameter selection.

## Tells that get you down-leveled

- Treating CV as universally applicable.
- Not knowing what time-series CV looks like.
- "Use 5-fold for everything."
- Confusing CV with bootstrap or with a held-out test set.

## Common follow-up

"How would you cross-validate a time series?"

The L6 answer:

> "Walk-forward (also called expanding-window or sliding-window) CV: train on data up to time t, test on (t, t+h]. Slide forward, repeat. Never sample test points from before training points. The fold count and window size are tuned to the autocorrelation length and the prediction horizon."

---

*Related: [Walk me through the bias-variance tradeoff](/questions/bias-variance-tradeoff/), [A/B testing for ML systems](/concepts/ab-testing-for-ml/).*
