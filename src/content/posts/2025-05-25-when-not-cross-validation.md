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
<figure class="learning-figure editorial-illustration" aria-labelledby="cross-validation-fit-the-question-edition-title"><span class="visual-fragment-anchor" id="cross-validation-question-title" aria-hidden="true"></span><p class="visual-title" id="cross-validation-fit-the-question-edition-title">A valid fold boundary follows the dependency and the deployment question</p><svg viewBox="0 0 360 628" role="img" aria-labelledby="cross-validation-fit-the-question-edition-svg-title cross-validation-fit-the-question-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="cross-validation-fit-the-question-edition-svg-title">A valid fold boundary follows the dependency and the deployment question</title><desc id="cross-validation-fit-the-question-edition-svg-desc">A time example trains Monday and Tuesday and tests Wednesday and Thursday; random training on Thursday while testing Tuesday leaks future information. A new-entity example trains A1/A2 and tests B1/B2; splitting A1/A2 across train/test leaks the entity. Model selection remains inside an untouched outer evaluation boundary. These repairs do not solve distribution shift.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Two dependency examples; deployment target still</tspan><tspan x="180" dy="19">matters</tspan></text><text x="18" y="70" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">TIME: training must precede evaluation</tspan></text><g data-editorial-node><rect x="25" y="102" width="62" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="56" y="128.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="56" dy="0">Mon</tspan><tspan x="56" dy="19">train</tspan></text></g><g data-editorial-node><rect x="108" y="102" width="62" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="139" y="128.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="139" dy="0">Tue</tspan><tspan x="139" dy="19">train</tspan></text></g><g data-editorial-node><rect x="191" y="102" width="62" height="62" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="222" y="128.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="222" dy="0">Wed</tspan><tspan x="222" dy="19">test</tspan></text></g><g data-editorial-node><rect x="274" y="102" width="62" height="62" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="305" y="128.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="305" dy="0">Thu</tspan><tspan x="305" dy="19">test</tspan></text></g><path d="M179 91L179 176" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="180" y="211" style="font-size:12px;font-weight:400;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">Random train Thu / test Tue crosses future -&gt;</tspan><tspan x="180" dy="19">past.</tspan></text><text x="18" y="267" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">GROUPS: evaluate unseen entities</tspan></text><g data-editorial-node><rect x="15" y="297" width="150" height="77" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="331" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Train: A1,A2</tspan><tspan x="90" dy="19">entity A only</tspan></text></g><g data-editorial-node><rect x="195" y="297" width="150" height="77" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="331" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Test: B1,B2</tspan><tspan x="270" dy="19">entity B only</tspan></text></g><text x="180" y="413" style="font-size:12px;font-weight:400;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">Random A1 train / A2 test leaks entity A across.</tspan></text><g data-editorial-node><rect x="20" y="460" width="320" height="108" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="481" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Model selection stays inside training:</tspan><tspan x="180" dy="19">inner tuning -&gt; freeze -&gt; untouched</tspan><tspan x="180" dy="19">outer test</tspan><tspan x="180" dy="19">never choose the model on its final</tspan><tspan x="180" dy="19">grade</tspan></text></g><text x="180" y="591" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Correct boundaries do not repair deployment</tspan><tspan x="180" dy="19">shift.</tspan></text></svg><figcaption><strong>Read it this way:</strong> The corrected boundaries protect time direction and unseen entities. Model selection must also stay inside the evaluation boundary. Dependency-safe folds still do not guarantee deployment representativeness. <span class="visual-reader-check" data-reader-check="cross-validation-fit-the-question"><strong>Check:</strong> Which dependency makes the random split invalid? Why does repairing that boundary not by itself fix deployment distribution shift?</span></figcaption></figure>

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
