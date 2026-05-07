---
title: "Implement KNN efficiently"
description: "The naive solution is one line. The interview is about scaling: when does naive fail, and what do you do?"
date: "2026-05-07"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: coding round at recsys, search, and embedding-team interviews.*

The L4 candidate writes the brute-force version. The L6 candidate names the regimes (small N, medium N, large N) and the right algorithm for each.

## The naive solution

```python
import numpy as np

def knn_brute(X, query, k):
    """X: (N, d) reference points. query: (d,). Returns top-k indices by L2 distance."""
    dists = np.linalg.norm(X - query, axis=1)
    return np.argpartition(dists, k)[:k]
```

Cost: O(N * d) per query, O(N) memory. Fine for N up to ~10K.

For batched queries, vectorize:

```python
def knn_brute_batch(X, queries, k):
    """X: (N, d), queries: (Q, d). Returns (Q, k) of indices."""
    dists = np.linalg.norm(X[None, :, :] - queries[:, None, :], axis=2)  # (Q, N)
    return np.argpartition(dists, k, axis=1)[:, :k]
```

Better: use the squared-distance trick to avoid the `sqrt`:

```python
# ||x - q||^2 = ||x||^2 + ||q||^2 - 2 * x.q
def knn_squared(X, queries, k):
    X_norm = (X**2).sum(axis=1)        # (N,)
    Q_norm = (queries**2).sum(axis=1)  # (Q,)
    dot = queries @ X.T                 # (Q, N)
    sq_dists = X_norm[None, :] + Q_norm[:, None] - 2 * dot
    return np.argpartition(sq_dists, k, axis=1)[:, :k]
```

This is the form used by FAISS's flat index. Order-of-magnitude faster than the naive form for batched queries because of BLAS-accelerated matmul.

## What an L5 answer should add

> "Brute-force scales as O(N * d) per query. For N up to ~100K and d ~ 100, brute-force on GPU with batched queries is fast enough.
>
> Past that, use approximate nearest neighbor (ANN):
>
> - **Tree methods** (KD-tree, ball tree): work well for low d (up to ~20). Useless for high d (curse of dimensionality).
> - **HNSW**: hierarchical navigable small world graphs. State-of-the-art for moderate-d (~100-1000). Sublinear query time, high recall. Used in Qdrant, Weaviate, pgvector with HNSW.
> - **IVF (inverted file)**: cluster the index, search only nearest clusters. Lower recall but very memory efficient. Used in FAISS.
> - **PQ (product quantization)**: compress vectors to bytes, do approximate distance computation. Trades recall for memory dramatically.
> - **HNSW + PQ**: best of both worlds for billion-scale.
>
> Trade-off knobs: recall vs query latency vs memory vs build time. Tune to your use case."

## What an L6 answer adds

> "...practical things:
>
> **Distance metric matters for index choice.** Cosine similarity, dot product, and L2 are different but related. For normalized vectors, cosine and L2 give the same ranking. For unnormalized, they differ. Some indexes (like inner-product HNSW) require careful normalization to be correct.
>
> **GPU vs CPU.** For very high-throughput query workloads, GPU brute-force (FAISS-GPU, ScaNN) is competitive with CPU ANN up to surprisingly large N (millions). For low-throughput or memory-constrained, CPU ANN wins.
>
> **Index updates are expensive.** Most ANN indexes are built once; adding/removing items is slow or unsupported. For dynamic catalogs, either rebuild periodically or use indexes with explicit incremental support (newer Qdrant, Vespa).
>
> **Recall is task-dependent.** 95% recall@10 is fine for recommendation; 99% is needed for some search applications. Tune the index parameters (HNSW M / efConstruction / efSearch, IVF nprobe) to your recall floor.
>
> **Don't roll your own.** FAISS, Qdrant, ScaNN, and Annoy cover the design space. Use them. The implementation details (cache-aware data layouts, SIMD-friendly distance computation) are decade-tuned."

## Tells that get you a strong-hire vote

- You name the **squared-distance + matmul trick** for batched queries.
- You distinguish **brute-force regimes** from **ANN regimes** by N.
- You name **HNSW, IVF, PQ** as the standard ANN families.
- You mention **recall vs latency vs memory** trade-offs.
- You **don't suggest implementing ANN from scratch**.

## Tells that get you down-leveled

- Loops over examples in Python (use vectorized ops).
- Computing `sqrt` when squared distance gives the same ranking.
- "Use a KD-tree" for high-dimensional data.
- No knowledge of HNSW.
- Confusing exact KNN with ANN.

## Common follow-up

"What if you have a billion vectors?"

The L6 answer:

> "Single-machine memory is your floor. A billion 768-dim float32 vectors is 3 TB; doesn't fit on one machine. Options: (1) Quantize aggressively (INT8 or PQ) to fit in memory. (2) Distribute across machines, query in parallel, merge top-K. (3) Use a hierarchical index where the top-level shards by cluster and only relevant shards are queried (IVF-style). Vespa, ScaNN, and large-scale FAISS deployments handle this. Don't build it from scratch; the engineering is more involved than the algorithm."

---

*Related: [Two-tower vs cross-encoder: when to use which?](/interviews/two-tower-vs-cross-encoder/), [Designing a RAG system that actually works](/essays/designing-rag-that-works/), [System design case study: personalized search ranking](/system-design/personalized-search-ranking/).*
