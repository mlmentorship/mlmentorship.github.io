---
title: "Approximate nearest neighbors: HNSW, IVF, and product quantization"
description: "Exact k-NN over a billion vectors is infeasible. ANN trades a small recall hit for a 100x to 10,000x speedup. The reason vector search at scale exists."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Approximate Nearest Neighbors** finds vectors close to a query without comparing against every stored vector. The three dominant methods are graph-based (HNSW), inverted-file (IVF), and quantization-based (PQ, OPQ). Production systems combine them.

## Why it matters

Modern embedding models produce 768- to 4096-dimensional vectors. Brute-force k-NN over $N$ vectors is $O(N \cdot d)$ per query. At $N = 10^9$ and $d = 1024$, that is roughly a teraflop per query, infeasible at retrieval QPS.

ANN delivers a tunable accuracy-speed-memory tradeoff. Modern vector databases (Faiss, Vespa, Milvus, Pinecone, Weaviate, pgvector) all use one or more of HNSW, IVF, and PQ. RAG, recommender retrieval, image search, and semantic search at scale would not exist without ANN.

## The three families

### HNSW (Hierarchical Navigable Small World)

A multi-layer proximity graph ([Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)).

- Each node is connected to its closest neighbors at each layer.
- Top layers are sparse (long-range jumps), bottom layer is dense.
- Search: enter at the top, greedily walk toward the query, drop down a layer, repeat.

| | |
|---|---|
| Recall@10 at typical config | 95 to 99 percent |
| Memory | 1.5 to 2x the raw vectors |
| Build time | Slow (graph construction is iterative) |
| Insert | Supported, online |
| Best for | High-recall, latency-critical, in-memory deployments |

### IVF (Inverted File Index)

Cluster the vectors with k-means into $K$ partitions ([Jégou et al., 2011](https://hal.inria.fr/inria-00514462/document)). At query time, find the nearest $\text{nprobe}$ centroids and only compare against vectors in those partitions.

| | |
|---|---|
| Recall trade-off | Tunable via $\text{nprobe}$ |
| Memory | Same as raw vectors |
| Build time | Fast (one k-means) |
| Best for | Disk-backed indexes, easy to shard |

### PQ (Product Quantization)

Compress each vector by splitting it into $m$ subvectors of dimension $d / m$ and quantizing each subvector independently to one of $2^b$ codewords ([Jégou et al., 2011](https://hal.inria.fr/inria-00514462/document)).

A 1024-dim float32 vector (4096 bytes) becomes $m$ codes of $b$ bits each. With $m = 64, b = 8$: 64 bytes. 64x memory reduction.

Distance computation uses precomputed lookup tables: for a query, precompute the squared distance from each query subvector to each codebook entry; the distance to a stored vector is the sum of $m$ table lookups.

| | |
|---|---|
| Memory | 32x to 128x reduction |
| Recall | Lower than HNSW; depends on codebook quality |
| Best for | Memory-constrained scale, billion-vector indexes |

### OPQ (Optimized PQ)

Apply a learned rotation before PQ to align variance with subvector boundaries. Strictly better than PQ at the same code budget.

## Production combinations

Real systems combine these:

- **IVF + PQ**: cluster, then quantize each partition's vectors. The Faiss workhorse.
- **HNSW + PQ**: HNSW graph for routing, PQ-compressed vectors for distance. Trades a recall hit for major memory savings.
- **HNSW + reranking**: use HNSW to get the top 100 candidates, rerank with the exact float vectors.
- **IVF-HNSW**: HNSW within each IVF partition.

## Recall vs latency vs memory

ANN has three knobs:

- **Recall@k**: fraction of true k-NN found.
- **Latency**: query time.
- **Memory**: bytes per vector.

You pick two. Examples:

- HNSW with high $efSearch$: high recall, low latency, high memory.
- IVF-PQ with small $\text{nprobe}$ and aggressive PQ: low memory, decent latency, lower recall.
- Brute force: 100 percent recall, infinite cost.

## Common pitfalls

- **Reporting recall without specifying k.** Recall@1 and recall@100 measure different failure modes.
- **Confusing index size with vector size.** HNSW adds graph overhead; PQ subtracts vector size.
- **Building once on cold data.** Most systems need incremental insert and delete; pick an index that supports them (HNSW, not raw IVF-PQ).
- **Over-indexing precision.** For semantic search, the embedding model is usually the noisier component; chasing 99.5 percent ANN recall is wasted effort if the embeddings have 90 percent retrieval precision.

## Related

- [Two-tower retrieval](/concepts/two-tower-retrieval/).
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity-metrics/).
- [RAG: retrieval-augmented generation](/concepts/rag-retrieval-augmented-generation/).
