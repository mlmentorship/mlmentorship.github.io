---
title: "When would you not use cross-validation?"
description: "Cross-validation is a tool, not a default. The senior answer names the cases where it's wrong, expensive, or misleading."
date: "2025-05-25"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at every level.*

The interviewer is checking whether you treat cross-validation as a reflex or as a choice. The L4 candidate uses it everywhere. The L6 candidate names specific cases where it's the wrong tool.

<p class="visual-kicker">Learning objective</p>
<p class="visual-title">Choose an evaluation that matches the deployment question before choosing folds.</p>

<!-- visual:cross-validation-fit-the-question -->
```mermaid
flowchart TB
	accTitle: Decide when ordinary cross-validation answers the wrong question
	accDescr: Start by asking whether the available labelled data represents the deployment population. If not, use a held-out production sample, domain-expert set, or online experiment instead of cross-validation alone. If it does, ask whether random folds preserve time order, groups, and spatial dependence. If not, use structure-aware folds. If observations are exchangeable, ask whether repeated model fits are affordable; use an ordinary cross-validation estimate when they are and a representative holdout when they are not. For any resampling score used to choose hyperparameters, tune only inside the evaluation split and reserve an outer fold or untouched test set. Branch labels and method names carry all meaning without color.
	Q{"Does labelled data represent<br/>the deployment population?"}
	E["USE EXTERNAL EVALUATION<br/>production sample, expert set, or A/B test"]
	D{"Would random folds preserve<br/>time, groups, and spatial dependence?"}
	S["USE STRUCTURE-AWARE FOLDS<br/>walk-forward, grouped, or spatial blocks"]
	C{"Can you afford<br/>repeated model fits?"}
	H["USE ONE REPRESENTATIVE HOLDOUT"]
	K["USE ORDINARY CROSS-VALIDATION"]
	T{"Will these scores choose<br/>hyperparameters or models?"}
	N["SEPARATE SELECTION FROM EVALUATION<br/>tune inside; reserve outer fold or final test"]
	R["REPORT THE RESAMPLED ESTIMATE"]
	Q -->|"no: distribution shift"| E
	Q -->|"yes"| D
	D -->|"no: dependent rows"| S
	D -->|"yes: exchangeable rows"| C
	C -->|"no"| H
	C -->|"yes"| K
	S --> T
	K --> T
	T -->|"yes"| N
	T -->|"no"| R
	class Q,D,C,T viz-focus
	class E,H,N viz-warning
	class S,K viz-state
	class R viz-output
	class E,S,H,K,N,R viz-wide
```
<p class="diagram-caption"><strong>Read it this way:</strong> decide what must remain unseen before deciding how many folds to run. Distribution shift calls for evaluation on the target population, dependence calls for boundaries that keep future, groups, or nearby locations out of training, and prohibitive cost calls for one representative holdout. If a resampled score selects the model, keep an outer fold or final test untouched so selection does not grade itself. Original synthesis informed by <a href="https://scikit-learn.org/stable/modules/cross_validation.html">scikit-learn's cross-validation guide</a> and <a href="https://jmlr.org/papers/v11/cawley10a.html">Cawley and Talbot's analysis of selection bias</a>.</p>

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
> 3. **There's a separate, decisive evaluation signal.** If you have a held-out production traffic sample, an A/B test, or a domain-expert evaluation set, that signal beats CV on the training distribution.
>
> 4. **The dataset is large enough.** Past ~10K samples for tabular tasks, the variance reduction from CV is small compared to a single well-chosen split."

This is L5. You've named the cases concretely and given the right alternative for each.

## What an L6 answer sounds like

> "...a few subtler points:
>
> **CV doesn't fix distribution shift.** If your test data has a different distribution than your training data (which it usually does in production), CV on training data tells you about training-distribution generalization, not deployment performance.
>
> **CV can mislead model selection at scale.** With many candidate models and small CV variance, the best CV score is often a noisy lucky pick. Repeated CV (averaging over multiple random splits) reduces this. Bootstrap-CI on the CV difference is more honest than 'model A beat model B.'
>
> **Nested CV is needed for honest hyperparameter selection.** Picking hyperparameters with CV and reporting that same CV score as your generalization estimate is a form of leakage. Nested CV (outer loop for evaluation, inner loop for hyperparameter tuning) is the principled fix, but it's expensive and rarely done in practice. Most people use a separate held-out test set instead."

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
