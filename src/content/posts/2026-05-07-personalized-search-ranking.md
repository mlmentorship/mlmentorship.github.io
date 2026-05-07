---
title: "System design case study: building personalized search ranking"
description: "An end-to-end design of a personalized search ranking system at scale, from problem framing through deployment and monitoring. The same template works for most ML system design interviews."
date: "2026-05-07"
draft: false
tags: ["essays", "system-design"]
category: "essays"
---


A personalized search ranking system at scale, structured as: problem framing, architecture, evaluation, operations. The structure generalizes to most ML system design problems.

## 1. Framing the problem

Before any architecture, clarify:

- **What surface?** A consumer product's main search? A vertical search (jobs, videos, products)? An enterprise search? Each has very different scale, latency, and quality bars.
- **Who's the user?** B2C with millions of users? B2B with thousands of accounts? Each user with a clear context, or strangers querying once?
- **What does "good ranking" mean?** Click-through rate? Long-term engagement? Task completion? User-stated satisfaction? These point to different model architectures and labels.
- **What's the scale?** Number of items in the catalog (thousands? billions?), queries per second, latency budget (10ms? 100ms? 500ms?), cost budget.
- **What's the personalization signal?** Do we have user history? Demographics? Real-time context? How much can we use without privacy issues?

For this case study, assume:
- A consumer streaming product (like Netflix or Spotify), main search bar.
- Catalog: ~10M items.
- Users: ~100M MAU.
- Queries: ~10K QPS.
- Latency: p99 &lt; 200ms (search needs to feel instant).
- Optimizing for: long-term engagement (a combination of immediate clicks, watch time, and next-week return).

## 2. Top-level architecture

Search ranking uses a **multi-stage retrieve-then-rank pipeline**. Why:

- Scoring 10M candidates per query at production latency is infeasible with anything but the simplest models.
- The top 10 results matter; positions 100-10M are noise.
- The information you can extract from a query is limited; for 99% of items, "this is irrelevant" can be decided cheaply. For the top 1000, you want to invest more compute.

So:

```
Query → [Stage 1: Candidate Generation] → ~5K candidates →
        [Stage 2: First-pass Ranking]    → ~500 candidates →
        [Stage 3: Final Ranking]         → ~50 candidates →
        [Stage 4: Re-ranking / Diversification] → top 10 → User
```

Each stage uses a more expensive model on a smaller set of candidates. Total budget: ~150ms.

## 3. Stage 1: Candidate generation (~5ms budget)

Reduce 10M items to ~5K. Multiple parallel sources:

### Lexical retrieval (BM25)
- Standard inverted index over titles, descriptions, transcripts, metadata.
- Handles exact-match queries, named entities, error messages.
- Critical for queries where embeddings underperform (rare terms, out-of-distribution words, exact title lookups).

### Embedding-based retrieval (two-tower)
- Pre-compute embeddings for all items via a content tower (text + metadata + popularity features).
- At query time, compute query embedding via a query tower (query text + user features).
- Approximate nearest-neighbor lookup (HNSW, Scann) in 1-5ms over the full catalog.
- Captures semantic similarity that BM25 misses.

### User-personalized retrieval
- For users with history: sequence model over recent watches/clicks generates a "user state" embedding.
- ANN lookup against item embeddings to retrieve "items similar to your recent activity."
- Independent of the query, generates a candidate set that biases toward what this user is likely to want.

### Trending / popularity
- Top-K most-clicked items overall and per-segment (genre, region, demographic).
- Catches what's hot regardless of personalization.

**Hybrid combination**: union the candidate lists from each source. ~5K candidates total. Order doesn't matter at this stage; the next stage will re-score.

### Why this design
- Each source has distinct failure modes. Combining them reduces tail failure (queries where one source returns nothing usable).
- Pre-computed embeddings + ANN gives sub-10ms scaling to billions of items.
- Lexical retrieval is fast and handles cases embeddings don't.
- User-personalized retrieval brings in items the user might want even if they didn't query for them precisely.

## 4. Stage 2: First-pass ranking (~30ms budget)

Score 5K candidates with a moderately-expensive model. Goal: cut to ~500.

A typical model here:
- **Architecture**: a lightweight neural ranker (DCN-V2, or a small transformer over feature embeddings).
- **Features**: query text, item metadata, user features, retrieval signals (which sources returned this item with what scores), recent interaction context.
- **Output**: a single relevance score per (query, item) pair.
- **Inference**: batched scoring, ~30ms for 5K candidates.

This stage doesn't need to be perfect, it just needs to keep the right ~500 in the top-500. Recall@500 is the metric to optimize here, not NDCG.

## 5. Stage 3: Final ranking (~50ms budget)

Score 500 candidates with a more expensive model. Goal: cut to ~50.

Differences from stage 2:
- Larger model with more capacity and richer features.
- Multi-task: predict multiple targets simultaneously (immediate click probability, watch duration, completion probability, next-day return probability).
- Combine multiple targets into a final score, possibly with calibration.

Modern recipe: a transformer over (query, user_history, candidate_features), producing per-task scores combined into a final ranking score.

## 6. Stage 4: Re-ranking and diversification (~20ms budget)

The top 50 from stage 3 are *individually* good but might form a bad list. A re-ranking step ensures:
- **Diversity**: not 10 results from the same artist or franchise.
- **Calibrated representation**: the right mix of categories given the query.
- **Freshness**: some boost for recent content.
- **Business rules**: licensing, regional availability, content-policy constraints.

Algorithm: typically a small heuristic re-rank (DPP-based diversity, or a simple max-marginal-relevance), followed by hard rule filtering. Output: top 10 results.

## 7. Labels and training

The trickiest part of search ranking is what to predict. Choices:

### Implicit feedback (clicks, watches)
- Abundant but heavily biased: we only see what we showed. Result: a model trained on observational data can learn the existing system's biases.
- Mitigation: inverse propensity weighting (IPW) on examined positions, counterfactual policy evaluation.

### Explicit feedback (ratings, surveys)
- High quality but very sparse.
- Use as auxiliary training signal or for evaluation.

### Long-term outcomes (next-day return, subscription retention)
- Most aligned with business but hardest to predict directly.
- Multi-task: predict short-term proxies, combine using long-term-aware weighting.

A common production recipe:
- Multi-task ranker predicts click, watch_complete, watch_30s, next_day_return, etc.
- Combine into a final score: `final = sum(w_i * P(task_i))` with weights tuned offline by counterfactual eval and confirmed in A/B.

## 8. Cold-start

Two cold-start problems:

### New users (no history)
- Strategy: start with popularity / segment-based rankings. Apply minimal personalization.
- Quickly accumulate signal in the first few sessions; transition to full personalization after ~10 interactions.

### New items (no engagement signal)
- Strategy: rely on content-based features (title, description, embedding from text/audio/visual).
- Boost new items in early hours to gather initial signal.
- After enough interactions accumulate, the model uses both content and engagement signals.

## 9. Evaluation

### Offline metrics
- **NDCG@K, MRR, Recall@K** on a held-out set of (query, ideal item) pairs.
- **Counterfactual estimators** (IPS, doubly robust) for unbiased estimates on logged data.
- **Caveat**: offline metrics on observational data are unreliable. Treat as sanity checks, not as decision metrics.

### Online metrics
- **Primary**: long-term satisfaction proxy (next-week return rate).
- **Secondary**: immediate engagement (CTR, watch time on retrieved items).
- **Guardrails**: latency p99, cost per query, fairness slices (per-creator-type, per-region, per-demographic), diversity metrics.
- **A/B testing**: minimum-detectable-effect calculation upfront; pre-committed sample size; multiple-comparison correction across slices.

### Human evaluation
- Quarterly rated relevance evaluations by trained raters on sampled queries.
- Used to calibrate the offline metrics and to detect systematic issues.

## 10. Serving infrastructure

### Real-time path
- Query understanding (spell correction, query rewriting): &lt;5ms.
- Stage 1 (candidate generation): ~5ms.
- Stage 2 (first-pass ranking): ~30ms.
- Stage 3 (final ranking): ~50ms.
- Stage 4 (re-ranking): ~20ms.
- **Total**: ~110ms; p99 budget of 200ms gives ~90ms headroom.

### Feature store
- User features computed in batch (daily) plus real-time updates for the most recent activity.
- Item features computed in batch when content is added or updated.
- Query features computed at request time.

### Index management
- Item embeddings re-computed weekly (or on content update).
- ANN index rebuilt nightly.
- Lexical index updated continuously as items are added.

### Model deployment
- Models versioned with full lineage (training data hash, code hash, hyperparameters).
- Shadow deployment for new models: run alongside production, compare predictions, no user impact.
- Gradual rollout: 1% → 10% → 50% → 100% with monitoring at each step.
- Automatic rollback on guardrail regression.

## 11. Monitoring

What to monitor:

### Quality
- Online primary metric and guardrails (compared against control).
- Offline-online correlation (is the offline eval predicting online correctly?).
- Failure-mode tracking: queries with no clicks, queries with regret (immediate query reformulation).

### System
- Latency per stage at p50 / p95 / p99.
- Cost per query.
- Cache hit rates.
- Index freshness.

### Distribution
- Query distribution shift (are users asking new things?).
- Content distribution shift (are new items being added at expected rates?).
- User behavior shift (is engagement pattern changing?).

### Failure modes
- Stage failures (timeout, exception): graceful degradation to simpler ranking.
- Empty result rate (queries returning &lt;5 results): often indicates a real bug.
- Latency spikes correlated with index updates or feature pipeline lag.

## 12. The hard problems

A few areas that are hard and tend to consume most of the senior judgment on a project like this:

### Feedback loops

Today's recommendations create tomorrow's training data. The system can self-reinforce a narrow distribution: items that get shown get clicked; clicks become training labels; the model learns to show those items more. Over time, diversity collapses.

Mitigations:
- Explicit diversity terms in re-ranking.
- Exploration in candidate generation (epsilon-greedy or Thompson sampling on a small fraction of impressions).
- Counterfactual augmentation: train the model on what *would have happened* if it had shown a wider distribution.

### Long-term vs short-term trade-offs

Optimizing immediate clicks can lead to clickbait, dissatisfaction, and long-term decline. Optimizing long-term metrics is hard because they're sparse and slow.

Mitigations:
- Multi-task with long-term proxies (next-week return, survey responses).
- Online experiments with long enough horizons to detect long-term effects (often 4-8 weeks).
- Periodic "diversity injection" to ensure the system doesn't trap itself.

### Calibration across heads

In multi-task ranking, each head is trained on different positive/negative ratios. Without explicit calibration, the noisiest head dominates the final score.

Mitigation: per-head calibration (Platt scaling or isotonic regression) on a held-out set before combining scores.

### Cold-start distribution

The "ideal" candidate generation is for the long tail (rare items / users), but those are exactly where the model has least signal. Disproportionate engineering effort goes into the cold-start tail relative to its share of traffic, but it's where competitive moat lives.

Mitigation: explicit content-based signals; treat cold-start as a separate model with separate training data (metadata-rich items only, with engagement signal artificially excluded).

## 13. Interview probing areas

If this came up in an interview, the interviewer would push on:

- **"Why two-stage and not one-stage?"**: latency math, recall vs precision trade-off.
- **"How do you train the candidate generator?"**: in-batch negatives + hard negatives + distillation from the full ranker.
- **"What's the eval bar for shipping?"**: primary metric significant lift, no guardrail regressions, no slice regressions in critical segments, novelty effect decayed.
- **"Tell me about a failure mode you'd worry about."**: feedback loops, calibration drift, distribution shift in either content or query distribution.
- **"What if your offline eval and online A/B disagree?"**: trust online; investigate why offline misled; rebuild offline eval on production-like distribution.

A senior candidate has clear answers to all of these. A staff candidate also discusses *organizational* concerns: how the team's metric definitions get aligned with product strategy; how model release cadence interacts with the eval pipeline; how to onboard new team members on this system.

## 14. What to build next (roadmap)

Even after the v1 ships:

- **v1.5**: better cold-start for new users (using content-based signals more aggressively in early sessions).
- **v2**: query understanding (LLM-powered query rewriting, intent classification, multi-step queries).
- **v2.5**: cross-surface personalization (use signals from related surfaces, what the user clicked on the home page, to inform search).
- **v3**: generative re-ranking (LLM that takes top 20 results and re-orders / generates explanations).
- **v3.5**: agentic search for complex queries (multi-step tool-using agent for queries that require synthesis across multiple items).

The specific roadmap depends on what's working and what isn't, but having a multi-quarter view of where the system is going is itself a senior signal.

---

This is the depth to aim for in a real senior interview. The interviewer doesn't need every detail; they need to see that you've actually thought about the system, not just listed its components.

If you can structure a 45-minute conversation around these 14 layers and answer follow-ups on any one of them, you're operating at the senior-staff bar.

---

*Related: [Design YouTube's recommender](/interviews/design-youtube-recommender/), [Two-tower vs cross-encoder: when to use which?](/interviews/two-tower-vs-cross-encoder/), [A/B testing for ML systems](/reference/ab-testing-for-ml/).*
