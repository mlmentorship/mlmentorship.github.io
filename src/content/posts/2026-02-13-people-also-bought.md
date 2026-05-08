---
title: "Design Amazon's people also bought"
description: "A simple-sounding feature with deep recsys ground underneath. The senior answer chooses between item-item collaborative filtering, embedding similarity, and learned co-purchase models, with explicit handling of feedback loops."
date: "2026-02-13"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: e-commerce and recsys interviews.*

The L4 candidate proposes "find similar items." The L6 candidate decomposes "similar" into specific behavioral signals (co-view, co-purchase, complementary) and reasons about which the surface should optimize for.

## What "people also bought" actually means

Three distinct signals, often confused:
1. **Co-purchase**: people who bought X also bought Y. Captures complementary items (printer + ink).
2. **Co-view**: people who viewed X also viewed Y. Captures substitutes (looking at multiple printers before deciding).
3. **Item similarity**: items with similar attributes / embeddings. Captures features (similar printers).

The product question: which one should the shelf optimize for? On a product page, you usually want *complementary* recommendations (you've already decided on this printer; what else might you need?), not substitutes (which would distract from the purchase). So co-purchase, not co-view, not item similarity.

## What an L5 answer sounds like

> "I'd build it as a candidate-generation + ranking system optimized for co-purchase:
>
> **Candidate generation**:
> - Item-item co-purchase matrix from order history. For item X, candidates are items frequently purchased in the same order or session.
> - Item-embedding similarity as a fallback for cold items.
> - Hybrid: union from both sources.
>
> **Ranking**: a small neural ranker scoring (anchor item, candidate, user, context) and predicting purchase probability. Multi-task: predict click, add-to-cart, purchase. Combined into a final score.
>
> **Eval**: offline (precision-at-K on held-out co-purchase pairs), online (CTR, add-to-cart-rate, attributed purchases on the recommendation surface, basket-size lift).
>
> **Cold-start**: new items have no co-purchase signal; rely on content-based candidates from item attributes / embeddings, boost in early period."

This is L5. Decomposed signals, two-stage architecture, eval framework.

## What an L6 answer adds

> "...practical things:
>
> **Feedback loops are aggressive here.** Today's recommendations create tomorrow's training data. Without exploration, the system collapses to recommending the most-purchased items everywhere. Mitigations: exploration in candidate gen, counterfactual augmentation, popularity discounting in the score (don't rank purely by raw co-purchase volume; that just amplifies the head).
>
> **Position-aware bias correction**: items in shelf position 1 get ~5x the engagement of position 5 regardless of relevance. Either model position explicitly when training the ranker, or randomize position for a small fraction of impressions for unbiased estimation.
>
> **Time decay matters.** Co-purchase from 5 years ago is less informative than co-purchase from last month. Exponential decay on the co-purchase weights, with the half-life tuned to the catalog churn rate.
>
> **Attribution is a measurement minefield.** When a user buys a recommended item, did the recommendation cause the purchase, or were they going to buy it anyway? Last-touch attribution overstates recsys impact; counterfactual measurement (showing the recommendation to half the traffic, no recommendation to the other half) is the only way to know.
>
> **Multi-objective: don't optimize for purchases alone.** Long-term metrics (return rate, customer lifetime value) sometimes prefer different recommendations than short-term purchase optimization. The product team has to set the weighting; the model learns to that weighting."

## Tells that get you a strong-hire vote

- You distinguish **co-purchase, co-view, and similarity** as different signals.
- You match the signal to the **product purpose** (complementary on PDP).
- You bring up **feedback loops** and exploration.
- You mention **counterfactual attribution** for measuring impact.
- You discuss **time decay** in the co-purchase signal.

## Tells that get you down-leveled

- "Use cosine similarity on item embeddings" without considering co-purchase signal.
- No mention of feedback loops.
- No attribution discussion.
- Treating "purchases caused by the model" as observable.

## Common follow-up

"What if the user has already bought X? How would you avoid recommending X?"

The L6 answer:

> "Two patterns: (1) hard exclusion of items the user has already bought in their account history, (2) soft de-ranking for items the user has viewed recently or has in cart (avoid recommending the obvious things they're already considering). Both need a user-state lookup at request time, which adds latency; cache user-purchased-set per user with TTL. The hard part is when 'already bought' is ambiguous (consumables they buy repeatedly: dog food, ink); use category-level rules for repeat-purchase signals."

---

*Related: [Design YouTube's recommender](/questions/design-youtube-recommender/), [Two-tower vs cross-encoder: when to use which?](/questions/two-tower-vs-cross-encoder/), [System design case study: personalized search ranking](/system-design/personalized-search-ranking/).*
