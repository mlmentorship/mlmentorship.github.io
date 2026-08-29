---
title: "Content-based filtering"
description: "Content-based filtering scores item features against a user profile. It handles item cold-start and often complements collaborative filtering."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Content-based filtering recommends items whose **features** (genre, text, tags, embeddings) match a **profile built from the items a user already engaged with**: it scores item–profile similarity, using **no other users' behavior**.

Content-based filtering is the standard answer to the **item cold-start** problem: a brand-new item with zero interactions has no collaborative signal, but it *does* have features, so a content model can recommend it on day one. It also drives **explainability** ("because you watched X") and works for niche users with unique tastes. Every recsys interview expects you to contrast it with collaborative filtering and explain why production systems combine them.

## The mechanism

1. **Item representation.** Turn each item into a feature vector: structured attributes (genre, brand, price), TF-IDF / embeddings of text, image/audio embeddings, or learned content encoders.
2. **User profile.** Aggregate the representations of items the user engaged with, e.g. the (weighted) average of liked-item vectors, or a learned user encoder.
3. **Score and rank.** Recommend items with the highest similarity (cosine / dot product) to the profile, excluding already-seen items.

This is structurally a **two-tower** idea (a user/profile tower and an item/content tower) when both sides are learned, which is why content features feed naturally into modern retrieval models.

## Content-based vs collaborative filtering

| | **Content-based** | **Collaborative filtering** |
| --- | --- | --- |
| Signal | item **features** + this user's history | the **user–item interaction matrix** |
| New item (item cold-start) | **works** (has features) | fails (no interactions) |
| New user | needs a little history | fails (no interactions) |
| Serendipity / discovery | weak (stays near known tastes → **filter bubble**) | strong (finds non-obvious patterns) |
| Niche users | strong | weak |
| Needs other users? | no | yes |
| Quality ceiling | limited by feature quality | learns latent taste it can't name |

The crisp summary: **content-based asks "what is this item like?"; collaborative asks "who else behaved like you?"** They fail in opposite situations, which is exactly why they're combined.

## Strengths and weaknesses

**Strengths**: handles item cold-start, needs no other users, recommendations are explainable, works for unique tastes.

**Weaknesses**:

- **Limited serendipity**: recommendations cluster around what the user already likes (the **filter-bubble / over-specialization** problem).
- **Feature-bound**: quality is capped by how good your item features are; it can't discover preferences your features don't encode.
- **Still has user cold-start**: a brand-new user with no history has no profile.

## Hybrid systems (what's actually deployed)

Production recommenders blend both:

- **Cold-start handoff**: content-based for new items/users, sliding to collaborative as interactions accumulate.
- **Feature-rich two-tower / wide-and-deep models** that take both content features *and* collaborative IDs as input, learning a single ranker.
- **Knowledge-graph and embedding side-information** layered onto collaborative factors.

So "content-based vs collaborative" is rarely a real either/or in 2026: the design question is *how* to fuse them.

## What an interviewer expects you to say

1. Define it as **profile (from the user's items) × item features**, with **no reliance on other users**.
2. Lead with its killer use case: **item cold-start** and **explainability**.
3. Contrast cleanly with collaborative filtering on the cold-start and serendipity axes ("what is this item like" vs "who behaves like you").
4. Name its weaknesses: **over-specialization / filter bubble**, **feature-quality ceiling**, and remaining **user cold-start**.
5. Conclude with **hybrid** systems and feature-rich two-tower models as the production reality.

## Common confusions

- **"Content-based solves all cold-start."** It solves **item** cold-start; a brand-new **user** still has no profile.
- **"It's just collaborative filtering with features."** It uses *no* cross-user signal; that's the defining difference and the source of both its cold-start strength and its serendipity weakness.
- **"It's more accurate than collaborative filtering."** Usually the opposite once interaction data exists: collaborative filtering learns latent preferences content features can't capture. Content shines specifically when behavioral data is sparse.
- **"Two-tower retrieval is collaborative filtering."** Two-tower can be either or both: with content features in the item tower it's content-based; with pure ID embeddings it's collaborative.

---

*Related: [Matrix factorization for recsys](/concepts/matrix-factorization-recsys/), [Two-tower retrieval](/concepts/two-tower-retrieval/), [How would you do cold-start for a new user?](/questions/cold-start-new-user/), [Knowledge-graph embeddings](/concepts/knowledge-graph-embeddings/), [TF-IDF and BM25](/concepts/tf-idf-and-bm25/).*
