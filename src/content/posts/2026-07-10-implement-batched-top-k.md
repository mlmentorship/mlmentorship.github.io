---
title: "Implement memory-bounded batched top-k retrieval"
description: "Retrieve top-k cosine neighbors for many queries without materializing the full query-by-item score matrix."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> Implement exact top-k cosine retrieval for query matrix `Q` and item matrix `X`. The full `Q @ X.T` matrix does not fit in memory.

This is the retrieval stage of an embedding system (two-tower recsys, semantic search, RAG) written by hand: score a batch of query embeddings against the item matrix and keep the top-k neighbors when the full score matrix will not fit in memory. It tests vectorization, numerical hygiene, and memory reasoning, not knowledge of a particular ANN library.

## Contract

```text
Q: float array [num_queries, dim]
X: float array [num_items, dim]
k: 1 <= k <= num_items
block_size: number of items scored at once
return: item indices and cosine scores [num_queries, k], sorted descending
```

Handle zero vectors explicitly. State whether they produce score zero or an error.

## Baseline reasoning

Normalize rows once, then process item blocks:

1. Score `Q_normalized @ X_block_normalized.T`.
2. Take each query's top-k within the block.
3. Merge those candidates with the running top-k.
4. Repeat until all items are consumed.

The running state is only `O(num_queries × k)`, and each score block is `O(num_queries × block_size)`.

## Reference implementation sketch

```python
import numpy as np


def row_normalize(a, eps=1e-12):
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norms, eps)


def exact_topk_cosine(q, x, k, block_size=4096):
    if q.ndim != 2 or x.ndim != 2 or q.shape[1] != x.shape[1]:
        raise ValueError("incompatible shapes")
    if not 1 <= k <= x.shape[0]:
        raise ValueError("invalid k")

    qn = row_normalize(q)
    best_scores = np.full((q.shape[0], k), -np.inf)
    best_indices = np.full((q.shape[0], k), -1, dtype=np.int64)

    for start in range(0, x.shape[0], block_size):
        stop = min(start + block_size, x.shape[0])
        scores = qn @ row_normalize(x[start:stop]).T
        local_k = min(k, stop - start)
        local = np.argpartition(scores, -local_k, axis=1)[:, -local_k:]
        local_scores = np.take_along_axis(scores, local, axis=1)
        local_indices = local + start

        merged_scores = np.concatenate([best_scores, local_scores], axis=1)
        merged_indices = np.concatenate([best_indices, local_indices], axis=1)
        keep = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
        best_scores = np.take_along_axis(merged_scores, keep, axis=1)
        best_indices = np.take_along_axis(merged_indices, keep, axis=1)

    order = np.argsort(-best_scores, axis=1)
    return (
        np.take_along_axis(best_indices, order, axis=1),
        np.take_along_axis(best_scores, order, axis=1),
    )
```

## Complexity

- Time: $O(QND)$ for exact scoring, plus top-k selection
- Peak score memory: $O(QB)$ for block size $B$
- Persistent result memory: $O(Qk)$

Blocking fixes memory, not exact-computation cost. For very large catalogs, discuss ANN indexing after producing a correct baseline.

## L4, L5, and L6 signals

- **L4:** correct simple implementation with shapes, tests, and complexity.
- **L5:** block processing, vectorization, edge cases, and benchmark plan.
- **L6:** distinguishes exact from ANN, reasons about host/device transfer, mixed precision, batching, index freshness, and quality/latency/cost trade-offs.

## Tests to write

- Compare against a full score matrix on a small random case.
- `k = 1`, `k = num_items`, and final partial block.
- Duplicate scores and deterministic ordering policy.
- Zero query and item vectors.
- One query, one item, and invalid shapes.

## Common mistakes

- Materializing the full score matrix despite the constraint.
- Normalizing inside the query loop.
- Using full sort instead of top-k selection.
- Returning unsorted final candidates.
- Claiming blocking changes asymptotic compute.

*Related: [approximate nearest neighbors](/concepts/approximate-nearest-neighbors/), [embedding similarity](/concepts/embedding-spaces-and-similarity/), and [two-tower retrieval](/concepts/two-tower-retrieval/).*
