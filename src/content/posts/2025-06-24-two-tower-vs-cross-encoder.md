---
title: "Two-tower vs cross-encoder: when to use which?"
description: "The recsys / search architecture decision that comes up in every retrieval interview. The right answer is 'both, in sequence.'"
date: "2025-06-24"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: recsys, search, and retrieval interviews.*

The textbook answer is "two-tower is faster, cross-encoder is more accurate." The senior answer is "both, in sequence: bi-encoder for retrieval, cross-encoder for reranking." Trade-offs and training discipline are where the level signal sits.

## What an L4 answer sounds like

> "Two-tower is faster but less accurate. Cross-encoder is more accurate but slower. So you use two-tower when speed matters and cross-encoder when accuracy matters."

Right idea, missing the architectural point. You've heard the comparison but haven't built either.

## What an L5 answer sounds like

> "These solve different problems and the right answer is usually 'both, in sequence.'
>
> **Two-tower** (also called bi-encoder, dual-encoder, or dual-tower) encodes query and item *independently* into a shared embedding space. At serving time, you compute query embeddings on demand, look up item embeddings from a precomputed index, and rank by dot product or cosine similarity. The advantage: item embeddings can be precomputed, so retrieval over a billion items is a sub-10ms approximate-nearest-neighbor lookup. The disadvantage: query and item embeddings are computed without seeing each other, so fine-grained interactions are missed.
>
> **Cross-encoder** takes (query, item) as a *joint* input, runs them through the same model, and outputs a relevance score. Strictly more expressive because it can model cross-attention between query and item tokens. The disadvantage: you have to score every (query, item) pair at serving time, which is O(N) per query, infeasible for large catalogs.
>
> **The standard production architecture is two-stage**: use a two-tower for *retrieval* (reduce billions to ~thousands of candidates), then a cross-encoder (or other reranker) for *ranking* (reorder the top thousands down to the final ~10). This gets you the speed of two-tower for the bulk of the work and the accuracy of cross-encoder where it matters.
>
> The cross-encoder reranker is typically the highest-ROI quality lever in any retrieval system, providing a meaningful precision gain on top-K for modest additional latency."

This is L5. You've named the architectures, the trade-offs, the standard composition, and the operational reality.

## What an L6 answer sounds like

> "...and a few more things that come up in practice:
>
> **Training data is different.** Two-tower is usually trained with contrastive objectives (in-batch negatives, hard negative mining), where the loss is symmetric and the negative samples have outsize impact on quality. Cross-encoder can be trained with simpler pointwise or pairwise losses because each (query, item) pair is independent.
>
> **The 'right' two-tower architecture has subtleties.** Modern variants include ColBERT (multi-vector representations, one embedding per token, late interaction at scoring time), which gets some of the accuracy of cross-encoder while keeping retrieval tractable; and SPLADE (sparse representations) for hybrid lexical-dense retrieval. These bridge the gap.
>
> **Hard negatives are everything for two-tower training.** In-batch negatives alone give you mediocre two-towers. The real wins come from mining hard negatives, items that are *almost* relevant. Common techniques: BM25-mined hard negatives, model-mined negatives from an earlier checkpoint, or curriculum negatives starting easy and getting harder.
>
> **Cross-encoder distillation back into two-tower** is a real lever. You can train your two-tower to mimic the cross-encoder's scores on a labeled set, gaining some of the cross-encoder's quality without its serving cost.
>
> **Late-interaction models** (ColBERT and similar) are increasingly common as a third option. Multi-vector retrieval, late interaction at scoring, a middle ground between two-tower and cross-encoder. More memory-intensive but better quality.
>
> **For RAG specifically**: the standard stack is bi-encoder retrieval (top 50-100) → cross-encoder rerank (top 5-10) → LLM generation. Each stage is its own engineering and quality optimization."

This is L6. You've gone past the textbook two-architecture comparison into the modern landscape (ColBERT, SPLADE, distillation) and the practical training discipline (hard negative mining).

## The tells that get you a strong-hire vote

- You frame it as **two stages, not two choices**.
- You bring up **hard negative mining** as the actual quality lever for two-tower.
- You mention **ColBERT or other late-interaction models** as a third architecture.
- You discuss **distillation** from cross-encoder to two-tower.
- You connect to **RAG**: this is the same architectural decision in disguise.

## The tells that get you down-leveled

- Treating it as a binary choice (one or the other).
- "Cross-encoder is slow" without quantifying ("would need to score every item per query, which is O(N)").
- No mention of reranking as a standard architectural step.
- No knowledge of training-data subtleties (hard negatives).
- Confusing two-tower with siamese networks (they're related but two-tower has different towers for query and item, often with different inputs).

## A common follow-up

"How would you train a good two-tower from scratch?"

The L6 answer:

> "Three phases:
>
> 1. **Warm-up with in-batch negatives**: train with simple contrastive loss where every other item in the batch is a negative. Gets you a reasonable starting point quickly.
> 2. **Hard negative mining**: use the warm-up model (or a separate strong model) to mine hard negatives, items that have high similarity score to the query but are not the gold positive. Train with these as negatives. This is where quality really comes from.
> 3. **Optional: distillation from a cross-encoder**: train a strong cross-encoder on a labeled set, score (query, candidate) pairs, then train the two-tower to match those scores. Distillation often outperforms direct supervised training.
>
> Throughout: be careful about the eval set. Standard MTEB benchmarks are public and over-fit; you want a domain-specific eval set built from production samples (or prospective production samples)."

If you can have this conversation fluently, you're at strong-senior depth.

---

*Related: [Designing a RAG system that actually works](/essays/designing-rag-that-works/), [Design YouTube's recommender](/interviews/design-youtube-recommender/).*
