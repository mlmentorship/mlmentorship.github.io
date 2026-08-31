---
title: "Implement KNN efficiently"
description: "The naive solution is one line. The interview is about scaling: when does naive fail, and what do you do?"
date: "2025-08-17"
draft: false
tags: ["questions"]
category: "questions"
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

**Learning objective:** choose an index by separating three decisions: whether approximation is allowed, how to avoid visiting every vector, and whether stored vectors must be compressed.

<!-- visual:knn-index-decision-funnel -->
<figure class="learning-figure plot-panel" aria-labelledby="knn-index-funnel-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="knn-index-funnel-title">Which constraint should change the KNN index?</p>
	<svg viewBox="0 0 360 478" role="img" aria-labelledby="knn-index-funnel-svg-title knn-index-funnel-svg-desc">
		<title id="knn-index-funnel-svg-title">A constraint-driven decision funnel for exact and approximate nearest-neighbor indexes</title>
		<desc id="knn-index-funnel-svg-desc">Start with a flat exact scan over all N vectors and batch it on the available hardware. Only if measured latency misses its target and approximation is allowed, choose a way to visit fewer vectors: KD or ball trees for low-dimensional data where partitions prune well, HNSW graph navigation for high-recall low-latency search when graph memory is acceptable, or IVF cluster probing when scan budget is controlled by nprobe. Product quantization is a separate optional lossy compression layer when vector memory or read bandwidth is the bottleneck. Every approximate path ends by measuring recall at k against flat ground truth together with latency, memory, build time, and update cost.</desc>
		<defs><marker id="knn-funnel-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<text class="viz-axis-label" x="18" y="22">1 · ESTABLISH THE EXACT BASELINE</text>
		<rect class="viz-node viz-node--input" x="47" y="34" width="266" height="56" rx="4"></rect>
		<text class="viz-callout" x="180" y="55" text-anchor="middle">FLAT · score all N vectors</text>
		<text class="viz-label" x="180" y="73" text-anchor="middle">exact · O(Nd) · batch / GEMM / GPU first</text>
		<path d="M180 90V120" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#knn-funnel-arrow)"></path>
		<text class="viz-label" x="188" y="108">misses latency SLO?</text>
		<rect class="viz-node viz-node--focus" x="51" y="122" width="258" height="48" rx="4"></rect>
		<text class="viz-callout" x="180" y="143" text-anchor="middle">Is approximation allowed?</text>
		<text class="viz-label" x="180" y="159" text-anchor="middle">if no, scale or shard the flat scan</text>
		<text class="viz-axis-label" x="18" y="197">2 · IF YES, VISIT FEWER VECTORS</text>
		<path d="M180 170V207M62 207H298M62 207V221M180 207V221M298 207V221" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
		<rect class="viz-node" x="10" y="222" width="104" height="82" rx="4"></rect>
		<text class="viz-callout" x="62" y="243" text-anchor="middle">KD / ball tree</text>
		<text class="viz-label" x="62" y="263" text-anchor="middle">low dimension;</text>
		<text class="viz-label" x="62" y="279" text-anchor="middle">partitions still</text>
		<text class="viz-label" x="62" y="295" text-anchor="middle">prune effectively</text>
		<rect class="viz-node" x="128" y="222" width="104" height="82" rx="4"></rect>
		<text class="viz-callout" x="180" y="243" text-anchor="middle">HNSW</text>
		<text class="viz-label" x="180" y="263" text-anchor="middle">navigate a graph;</text>
		<text class="viz-label" x="180" y="279" text-anchor="middle">fast, high recall;</text>
		<text class="viz-label" x="180" y="295" text-anchor="middle">graph costs RAM</text>
		<rect class="viz-node" x="246" y="222" width="104" height="82" rx="4"></rect>
		<text class="viz-callout" x="298" y="243" text-anchor="middle">IVF</text>
		<text class="viz-label" x="298" y="263" text-anchor="middle">probe selected</text>
		<text class="viz-label" x="298" y="279" text-anchor="middle">clusters; nprobe</text>
		<text class="viz-label" x="298" y="295" text-anchor="middle">sets scan budget</text>
		<text class="viz-axis-label" x="18" y="333">3 · COMPRESS ONLY IF MEMORY OR READS DOMINATE</text>
		<path d="M180 304V346" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8;stroke-dasharray:5 4;marker-end:url(#knn-funnel-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="47" y="348" width="266" height="55" rx="4" style="stroke-dasharray:5 4"></rect>
		<text class="viz-callout" x="180" y="369" text-anchor="middle">OPTIONAL PQ · store short lossy codes</text>
		<text class="viz-label" x="180" y="387" text-anchor="middle">fewer bytes · approximate distances</text>
		<path d="M180 403V426" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#knn-funnel-arrow)"></path>
		<rect class="viz-node viz-node--output" x="22" y="428" width="316" height="40" rx="4"></rect>
		<text class="viz-callout" x="180" y="446" text-anchor="middle">VERIFY AGAINST FLAT GROUND TRUTH</text>
		<text class="viz-label" x="180" y="461" text-anchor="middle">recall@k · latency · RAM · build · updates</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> start with the exact flat scan and optimize that baseline before buying index complexity. If it misses the measured latency target and approximation is acceptable, choose one candidate-pruning path for the data regime. Treat PQ as a separate compression decision; it can be combined with IVF or a graph index, but it introduces another source of distance error. Finally, compare recall@<var>k</var> with flat ground truth while measuring the operational costs. Original synthesis checked against the <a href="https://scikit-learn.org/stable/modules/neighbors.html">scikit-learn neighbors guide</a>, <a href="https://arxiv.org/abs/1603.09320">the HNSW paper</a>, and <a href="https://faiss.ai/">Faiss's research foundations</a>.</figcaption>
</figure>

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

*Related: [Two-tower vs cross-encoder: when to use which?](/questions/two-tower-vs-cross-encoder/), [Designing a RAG system that actually works](/guides/designing-rag-that-works/), [System design case study: personalized search ranking](/guides/personalized-search-ranking/).*
