---
title: "Design YouTube's recommender"
description: "The canonical recsys design question. The real test is whether you'll dive into model architecture or scope the problem first."
date: "2025-10-20"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: ML system design, especially recsys teams.*

The test is whether you can scope, decompose, and reason about trade-offs at scale, not whether you know the YouTube paper.

## The single biggest mistake

Diving straight into model architecture in the first 30 seconds. This is the number-one reason candidates get down-leveled on this question. A senior candidate spends the first 5-10 minutes asking questions and establishing scope before discussing any architecture.

## What an L4 answer sounds like

> "I'd use collaborative filtering with matrix factorization, then add content features. We can use a deep learning model with embeddings for users and videos. For ranking we can use gradient-boosted trees or a neural ranker. We'd serve it through a low-latency API."

Correct ingredients, no architecture, no scoping. You've memorized recsys components but haven't built one.

## What an L5 answer sounds like

A senior approach has structure. The L5 answer follows it:

### Step 1: Scope (5-10 minutes)

> "Before I propose an architecture, I'd want to clarify a few things:
>
> - **Surface**: are we talking about home page recommendations, watch-next, search ranking, or all of these? They have very different characteristics.
> - **Objective**: long-term watch time? Engagement? Subscriptions? User retention? These point to different model architectures and labels.
> - **User signal**: are we counting clicks, watch duration, completion rate, likes, shares? Each has different reliability and bias profile.
> - **Scale**: how many videos in the catalog? How many users? How many requests per second?
> - **Cold-start**: how do we handle new users and new videos? Can we use side information?
> - **Constraints**: latency budget for ranking? Cost? Existing infrastructure?
>
> Let me assume the home page surface, optimizing for long-term satisfaction (a combination of watch time, engagement, and signals like 'survey responses'), serving billions of users and a multi-billion video catalog at ~100ms p95."

The interviewer is now visibly relieved. You've signaled you can scope.

### Step 2: Two-stage architecture

> "At YouTube scale you can't score every video for every request. The standard architecture is two-stage:
>
> **Stage 1: Candidate generation**: reduce the catalog from billions to thousands.
> - Two-tower model with user and video embeddings, trained with sampled softmax or in-batch negatives.
> - User tower input: watch history (sequence of recently watched video embeddings, often via a sequence model), demographics, context (time, device).
> - Video tower input: video embedding (often pre-trained on co-watch or content), category, freshness.
> - Serve via approximate nearest neighbor (HNSW, Scann) with sub-10ms retrieval.
>
> **Stage 2: Ranking**: score the ~thousands of candidates more carefully.
> - A larger neural ranker (DCN-V2, deep cross network, or a transformer-based ranker).
> - Many more features: full user history, video metadata, contextual signals.
> - Multi-task learning: predict multiple targets simultaneously (watch probability, watch duration, like, comment) and combine in a final score.
> - Calibrate the score so it's interpretable as a probability."

### Step 3: Labels and training

> "The trickiest part of recsys is what to predict. A few choices:
>
> - **Implicit feedback** (clicks, watches) is abundant but biased, we only see what we showed. Need inverse propensity weighting or counterfactual reasoning to debias.
> - **Watch time** is a stronger signal than clicks but requires careful normalization (a 30-second click on a 30-second video is different from a 30-second click on a 30-minute video).
> - **Long-term outcomes** (next-day return, subscription) are most important but sparsest. Often modeled as auxiliary tasks.
>
> I'd predict multiple targets in a multi-task ranker, then combine them. Final score might be `f(P(watch), P(complete), P(like), P(survey_positive))` with weights tuned via offline counterfactual eval and confirmed in A/B."

### Step 4: Eval and serving

> "Offline eval: held-out user sessions, with metrics like NDCG, MRR, and counterfactual estimators (IPS, doubly robust). But offline metrics on observational data are unreliable for recsys; the *real* eval is the A/B test.
>
> Online: A/B test on a small fraction of traffic. Primary metric is long-term satisfaction proxy (e.g., next-week return rate); guardrails on engagement and on diversity / fairness slices.
>
> Serving: candidate generation in ~10ms (ANN lookup), ranking in ~50ms (batched scoring), final composition / re-ranking in ~10ms. Total p95 ~100ms."

This is L5. Structured, scope-aware, technically defensible. You'd be hired.

## What an L6 answer sounds like

The L6 answer is the L5 answer plus the things only people who've shipped at this scale know:

### Add: the calibration / multi-task subtleties

> "...one thing I'd want to flag is calibration. In a multi-task ranker, each head is trained on a different positive/negative ratio, clicks are dense, completions are sparser, surveys are very sparse. Without explicit calibration, the head with the noisiest probabilities will dominate the final score. I'd add a calibration step (Platt scaling or isotonic regression) per head before combining."

### Add: the feedback loop problem

> "...recsys has a strong **feedback loop**: today's recommendations create tomorrow's training data, so the system can self-reinforce a narrow distribution. I've seen teams ship a 'better' model offline that immediately tanked diversity online because it learned to recommend exactly what users had previously clicked. Mitigations: explicit diversity terms in the ranker, exploration in candidate generation (epsilon-greedy or Thompson sampling on the head distribution), counterfactual augmentation."

### Add: the cold-start strategy

> "...for new videos, the embedding-based two-tower has nothing to retrieve. I'd add a content-based candidate source (text/audio/visual features extracted from the video) running in parallel with the collaborative source, and explicitly boost new content in early hours. For new users, I'd start with popularity-based recommendations and switch to the personalized model after a few sessions of signal."

### Add: the long-term-vs-short-term trade-off

> "...the hardest open problem in recsys is reconciling short-term engagement (clicks, watch time today) with long-term satisfaction (does this user still come back in 6 months). Pure engagement optimization can lead to clickbait and dissatisfaction. The standard mitigation is to include long-term proxies (survey responses, return rate) in the multi-task objective, but the weights are hard to set and require careful A/B over long horizons. This is the area where I'd expect to spend most of my product judgment."

This is L6. You've gone past the architecture into the *operational hard problems* that real recsys teams spend most of their time on.

## The tells that get you a strong-hire vote

- You **scope before solving**. 5-10 minutes of clarifying questions before any architecture.
- You distinguish **candidate generation vs ranking** as a fundamental architectural decision.
- You bring up **feedback loops**: **calibration across tasks**: and **cold-start** unprompted.
- You acknowledge that **A/B tests are the real eval** and that offline metrics on observational data are unreliable.
- You discuss **what could go wrong in production** (clickbait, filter bubbles, distribution drift).

## The tells that get you down-leveled

- Diving into model architecture in the first minute.
- "I'd use a transformer for ranking" without justification.
- No mention of two-stage architecture (candidate gen + ranking).
- No mention of A/B testing or counterfactual eval.
- Treating the question as "list recsys techniques."
- Forgetting cold-start, diversity, and feedback loops.

## A common follow-up

"How would you measure if your recsys is causing filter bubbles?"

This is a senior probe. The L6 answer:

> "I'd track diversity at multiple levels: per-user diversity over time (entropy of recommended categories per user, week over week), system-level diversity (gini coefficient of impressions across the catalog), and demographic / content slice fairness (is one creator type systematically under-recommended). I'd also run periodic counterfactual analyses where I replay user sessions with diversity-augmented candidate sets to estimate the gap. The fundamental issue is that 'filter bubble' isn't precisely defined, you have to operationalize it as a specific metric you'll defend, and that operationalization is itself a product decision."

---

*Related: [What L5 vs L6 actually means at FAANG ML](/essays/l5-vs-l6-faang-ml/) for the level dynamics in system design specifically.*
