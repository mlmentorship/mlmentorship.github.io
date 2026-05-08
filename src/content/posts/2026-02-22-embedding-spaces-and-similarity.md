---
title: "Embedding spaces and similarity metrics"
description: "How learned vector representations encode meaning, and why cosine similarity is the default metric for retrieval and recsys."
date: "2026-02-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

An embedding space is a learned vector space in which points represent objects (words, sentences, images, items, users) and geometric relationships. Distance, angle. Encode semantic relationships such as similarity or relevance.

## Why it matters

Embeddings are the substrate for retrieval, recommendation, search, clustering, classification, and most LLM-adjacent products. A senior interview will check whether you can pick the right similarity metric, the right normalization, and the right index.

## Common similarity metrics

For two vectors $u, v \in \mathbb{R}^d$:

| Metric | Formula | When to use |
|--------|---------|-------------|
| **Dot product** | $u^\top v$ | When magnitudes are meaningful (e.g., learned matrix-factorization scores). Indexable with MIPS algorithms. |
| **Cosine** | $\frac{u^\top v}{\|u\|\,\|v\|}$ | Default for embeddings where direction encodes meaning and magnitude is a noise/popularity confound. |
| **Euclidean / L2** | $\|u - v\|_2$ | When distances have physical meaning (image patches in pixel space, geographic coordinates). |
| **Negative Euclidean²** | $-\|u-v\|^2$ | Equivalent to dot product on L2-normalized vectors plus a constant. |

Cosine is the default for sentence embeddings (BERT-family), CLIP, two-tower retrieval, and most embedding APIs. Reason: training objectives (contrastive, triplet) typically L2-normalize, making magnitude meaningless.

## L2 normalization

A common convention: project every embedding onto the unit sphere by dividing by its L2 norm before storing or comparing. Effects:

- Dot product equals cosine similarity (no extra division at query time).
- Vector index (FAISS, ScaNN, HNSW) can use Inner Product mode for cosine retrieval.
- Magnitude (which often correlates with item popularity or training frequency) is removed as a confound.

If you store unnormalized embeddings and compare with cosine, you're paying the normalization cost at every query.

## Geometry of learned embeddings

Empirical regularities in well-trained embedding spaces:

- **Clusters** form for semantically similar items.
- **Linear analogies** (king − man + woman ≈ queen) hold in word2vec / GloVe; less reliably in modern contextual embeddings.
- **Anisotropy**: contextual LM embeddings (BERT, GPT) often concentrate in a narrow cone; cosine on raw embeddings can be misleading. Whitening or mean-centering helps.
- **Curse of dimensionality**: in high-d, all pairwise distances concentrate. Distinguishing top-1 from top-10 becomes noisier. Useful embedding dimensions are typically 64–1024 even when the model space is much larger.

## Indexing for fast retrieval

For $N$ items and queries:

| Method | Build | Query | Recall |
|--------|-------|-------|--------|
| Brute force |. | $O(Nd)$ | exact |
| **HNSW** [(Malkov & Yashunin, 2018)](https://arxiv.org/abs/1603.09320) | $O(N \log N)$ | $O(\log N)$ | tunable, ~95–99% |
| **IVF + PQ** (FAISS) | $O(Nd)$ | $O(\sqrt{N})$ | tunable |
| **ScaNN** (Google) | $O(Nd)$ | $O(\log N)$ | tunable |

HNSW is the default for most production embedding stores (Pinecone, Weaviate, pgvector with hnsw index, Qdrant).

## Common pitfalls

- **Mixing normalized and unnormalized vectors in the same index.** Cosine and dot give different rankings.
- **Comparing across embedding models.** Vectors from BERT and CLIP live in unrelated spaces; concatenating or comparing across them is meaningless without alignment.
- **Treating embedding dimension as quality.** Higher-d embeddings are not strictly better; tradeoff is recall vs. storage and query latency.
- **Ignoring popularity bias.** Magnitude correlates with frequency; if you don't L2-normalize, popular items dominate top-k for everyone.
