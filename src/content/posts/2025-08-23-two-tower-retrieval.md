---
title: "Two-tower retrieval"
description: "Encode queries and items with separate networks into a shared embedding space; retrieve by approximate nearest neighbors. The default architecture for industrial recommenders and search."
date: "2025-08-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A two-tower model encodes the query and item with separate neural networks into a shared embedding space. A dot product or cosine scores each pair. Contrastive or sampled-softmax training pushes positive pairs above negatives.

Two-tower (a.k.a. dual-encoder) is the dominant architecture for the **retrieval** stage of large-scale ranking systems: web search, e-commerce search, YouTube recommendations, ad targeting, dense passage retrieval for RAG, semantic search.

The structural advantage: item embeddings can be precomputed once per item and indexed. At query time you only run the query tower (cheap) and do an approximate-nearest-neighbor lookup (sub-linear in catalog size). Cross-encoders (where query and item are concatenated into a single network) cannot be precomputed and are 100–10000× too slow for retrieval at scale.

## Architecture

```
query  → query_tower  → q ∈ R^d
item   → item_tower   → i ∈ R^d
score  = q · i  (or cosine)
```

- **Towers**: typically transformers, MLPs, or a mix. Towers usually do **not** share weights (different input modalities or feature sets).
- **Embedding dim $d$**: 64–512 in production. Higher $d$ is more expressive; lower $d$ is faster to index and more cache-friendly.
- **Output normalization**: L2-normalize so dot product equals cosine; lets the index use Inner Product mode (see [embedding spaces](/concepts/embedding-spaces-and-similarity/)).

## Training

Standard losses:

### In-batch sampled softmax
For a batch of $B$ positive (query, item) pairs, treat the other $B-1$ items in the batch as negatives. Loss per query:

$$
L = -\log \frac{\exp(q \cdot i^+)}{\sum_{j=1}^{B} \exp(q \cdot i_j)}.
$$

Cheap, parallelizes well, but biases toward popular items (popular items appear as negatives more often).

### Importance-corrected sampled softmax
Correct the in-batch sampling bias by subtracting $\log p(\text{item}_j)$ from each negative's logit. Standard in YouTube's two-tower [(Yi et al., 2019)](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/).

### Hard negative mining
Sample hard negatives (high-scoring but incorrect items) explicitly. More expensive but improves quality, especially after the model is past the easy-negatives stage.

## Two-stage architecture

In production systems, two-tower is almost always the **retrieval** stage, followed by a cross-encoder **ranker**:

1. **Retrieval (recall-oriented)**: two-tower returns top-K (e.g., 1000) candidates from millions of items in <10 ms via ANN.
2. **Ranking (precision-oriented)**: cross-encoder or feature-rich tree model ranks the K candidates with full feature interactions.

## Tradeoffs vs. cross-encoder

| Property | Two-tower | Cross-encoder |
|---------|-----------|---------------|
| Latency at scale | sub-linear (ANN) | linear in catalog |
| Quality | lower (no query-item interactions) | higher |
| Memory | one vector per item | none (recomputed per query) |
| Use case | retrieval | reranking |

## Common pitfalls

- **Using two-tower for ranking when accuracy matters.** Lacks fine-grained feature interactions.
- **Ignoring negative sampling bias.** In-batch sampled softmax favors popular items; always combine with importance correction or popularity de-biasing.
- **Forgetting to refresh item embeddings.** When the item tower changes (new training run), all item embeddings must be re-encoded and re-indexed. Plan for periodic offline re-embedding.
- **Comparing dot vs. cosine inconsistently.** Pick one (usually L2-normalized + dot) and use it everywhere.

## Related

- [Embedding spaces](/concepts/embedding-spaces-and-similarity/). Vector representations and indexing.
- [RAG overview](/concepts/rag-overview/). Retrieval-augmented generation uses two-tower for the retrieval step.
