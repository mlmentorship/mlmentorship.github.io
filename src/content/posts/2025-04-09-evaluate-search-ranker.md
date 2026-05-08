---
title: "How would you evaluate a search ranker?"
description: "Search ranking eval is offline metrics for development, A/B for shipping, and human raters for absolute calibration. The senior answer uses all three and respects what each measures."
date: "2025-04-09"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: search and information-retrieval interviews.*

The L4 candidate names NDCG. The L6 candidate combines offline metrics, online behavioral signals, and human relevance judgments, and knows what each is good for.

## What an L4 answer sounds like

> "Use NDCG@10 on a held-out set of queries with relevance labels."

Right tool, only one of three. You've heard the metric, not the eval system.

## What an L5 answer sounds like

> "Three layers, each measuring something different:
>
> 1. **Offline ranking metrics on a labeled set.** NDCG@K (graded relevance), MRR (first relevant), Recall@K (did we get the right docs at all). Used to compare model variants quickly during development. Cheap to compute, but only as good as the labels.
>
> 2. **Online behavioral metrics.** CTR at position, clicks-per-session, abandoned queries (no clicks), regret signals (immediate query reformulation). Cheap, but biased by position (top results get clicked more regardless of relevance) and by what the user already saw.
>
> 3. **Human relevance judgments.** Trained raters score (query, document) pairs against a rubric. Slow and expensive but the most calibrated signal. Used for absolute quality measurement and to train the offline labeled set.
>
> Pipeline: develop with offline metrics, ship with online A/B tests, calibrate with periodic human evaluations."

This is L5. Three layers, what each is good for.

## What an L6 answer adds

> "...practical points:
>
> **Position bias dominates online click metrics.** Position 1 gets ~5x the CTR of position 5 regardless of relevance. Mitigations: position-aware models (predict CTR conditional on position, then estimate relevance), randomized swaps for unbiased estimation (interleaving experiments), or click-model-based corrections (cascade model, dynamic Bayesian network).
>
> **Interleaving experiments are more powerful than A/B for ranker comparisons.** Show a user a mix of results from ranker A and ranker B; observe which they click. Per-query signal, much higher statistical power than user-level A/B.
>
> **Counterfactual eval on logged data.** With proper inverse propensity weighting, you can estimate how a new ranker would have performed on past traffic. Cheap pre-screening before launching a real A/B.
>
> **Slice-based evaluation is mandatory.** Aggregate metrics hide failures on important slices: rare queries (the long tail), specific verticals (jobs, news), specific user segments. A 1% NDCG lift overall while regressing 5% on the most valuable query class is a bad ship.
>
> **Human eval calibration drift.** Human raters disagree (inter-rater reliability ~0.6-0.7 for nuanced judgments). Re-train the rater pool periodically; track inter-rater agreement; treat absolute scores cautiously, deltas more reliably."

## Tells that get you a strong-hire vote

- You name **three layers** (offline / online / human).
- You bring up **position bias** as the dominant confound in online metrics.
- You mention **interleaving** for ranker comparison.
- You insist on **slice-based eval**.
- You acknowledge **human eval drift** and inter-rater reliability.

## Tells that get you down-leveled

- NDCG only.
- No mention of online metrics.
- No awareness of position bias.
- No slicing.

## Common follow-up

"How would you build a labeled ranking dataset?"

The L6 answer:

> "Sample queries from production traffic stratified by query frequency (head, torso, tail). For each query, collect candidate documents from production retrieval. Have trained raters score each (query, document) pair on a graded relevance scale (e.g., 0-4) using a documented rubric. Track inter-rater agreement and re-rate ambiguous cases. Refresh the dataset quarterly; production query distribution drifts faster than people expect. The rubric design is the most important step: vague rubrics produce noisy labels that look like measurement uncertainty but are spec uncertainty."

---

*Related: [Design YouTube's recommender](/questions/design-youtube-recommender/), [Two-tower vs cross-encoder: when to use which?](/questions/two-tower-vs-cross-encoder/), [A/B testing for ML systems](/concepts/ab-testing-for-ml/), [System design case study: personalized search ranking](/system-design/personalized-search-ranking/).*
