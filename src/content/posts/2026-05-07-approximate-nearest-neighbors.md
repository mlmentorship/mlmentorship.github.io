---
title: "Approximate nearest neighbors: HNSW, IVF, and product quantization"
description: "Exact k-NN over a billion vectors is infeasible. ANN trades a small recall hit for a 100x to 10,000x speedup. The reason vector search at scale exists."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Approximate Nearest Neighbors** finds vectors close to a query without comparing against every stored vector. The three dominant methods are graph-based (HNSW), inverted-file (IVF), and quantization-based (PQ, OPQ). Production systems combine them.

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

<!-- visual:ivf-pq-query-funnel -->
<figure class="learning-figure" aria-labelledby="ivf-pq-funnel-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ivf-pq-funnel-title">How do IVF and PQ remove different parts of the search cost?</p>
	<div class="visual-grid--two" role="group" aria-label="Two-stage IVF and product-quantization query funnel">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 220" role="img" aria-labelledby="ivf-route-title ivf-route-desc">
				<title id="ivf-route-title">IVF selects two of four vector partitions</title>
				<desc id="ivf-route-desc">A diamond query Q is closest to centroids C1 and C2. With nprobe equal to two, their solid partitions are scanned. The dashed C3 and C4 partitions are skipped, reducing how many stored vectors become candidates.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="185" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">1 · IVF NARROWS WHERE TO SEARCH</text>
				<ellipse cx="90" cy="91" rx="60" ry="42" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></ellipse>
				<ellipse cx="190" cy="137" rx="66" ry="49" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></ellipse>
				<ellipse cx="223" cy="55" rx="46" ry="25" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-dasharray:6 4"></ellipse>
				<ellipse cx="71" cy="174" rx="43" ry="24" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-dasharray:6 4"></ellipse>
				<circle cx="58" cy="78" r="3" style="fill:var(--viz-input-stroke)"></circle>
				<circle cx="83" cy="108" r="3" style="fill:var(--viz-input-stroke)"></circle>
				<circle cx="113" cy="70" r="3" style="fill:var(--viz-input-stroke)"></circle>
				<circle cx="158" cy="132" r="3" style="fill:var(--viz-focus-stroke)"></circle>
				<circle cx="196" cy="161" r="3" style="fill:var(--viz-focus-stroke)"></circle>
				<circle cx="225" cy="119" r="3" style="fill:var(--viz-focus-stroke)"></circle>
				<circle cx="211" cy="50" r="3" style="fill:var(--viz-neutral-stroke)"></circle>
				<circle cx="238" cy="61" r="3" style="fill:var(--viz-neutral-stroke)"></circle>
				<circle cx="61" cy="178" r="3" style="fill:var(--viz-neutral-stroke)"></circle>
				<circle cx="84" cy="166" r="3" style="fill:var(--viz-neutral-stroke)"></circle>
				<circle cx="96" cy="92" r="6" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:3"></circle>
				<circle cx="188" cy="137" r="6" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:3"></circle>
				<circle cx="223" cy="55" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:2"></circle>
				<circle cx="71" cy="174" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:2"></circle>
				<path d="M139 88L147 96L139 104L131 96Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></path>
				<path d="M136 100L105 91M144 101L179 130" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<text class="viz-callout" x="138" y="82" text-anchor="middle">query Q</text>
				<text class="viz-axis-label" x="76" y="48">C1 · SCAN</text>
				<text class="viz-axis-label" x="184" y="201">C2 · SCAN</text>
				<text class="viz-label" x="231" y="90">C3 · skip</text>
				<text class="viz-label" x="26" y="207">C4 · skip</text>
				<text class="viz-callout" x="151" y="117" text-anchor="middle">nprobe = 2</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 220" role="img" aria-labelledby="pq-distance-title pq-distance-desc">
				<title id="pq-distance-title">PQ scores each selected candidate with lookup tables</title>
				<desc id="pq-distance-desc">One selected candidate is stored as four codebook identifiers rather than its full vector. For query Q, four precomputed subvector-to-codeword distances, 0.04, 0.31, 0.08, and 0.22, are looked up and summed to an approximate distance of 0.65.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="185" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">2 · PQ MAKES EACH SCORE CHEAPER</text>
				<text class="viz-callout" x="20" y="50">candidate code</text>
				<rect x="20" y="59" width="58" height="39" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
				<rect x="78" y="59" width="58" height="39" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
				<rect x="136" y="59" width="58" height="39" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
				<rect x="194" y="59" width="58" height="39" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
				<text class="viz-callout" x="49" y="83" text-anchor="middle">2</text>
				<text class="viz-callout" x="107" y="83" text-anchor="middle">7</text>
				<text class="viz-callout" x="165" y="83" text-anchor="middle">1</text>
				<text class="viz-callout" x="223" y="83" text-anchor="middle">5</text>
				<text class="viz-label" x="272" y="83" text-anchor="middle">IDs</text>
				<path d="M49 101V120M107 101V120M165 101V120M223 101V120" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<text class="viz-label" x="20" y="116">query lookup</text>
				<rect x="20" y="123" width="58" height="34" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect>
				<rect x="78" y="123" width="58" height="34" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect>
				<rect x="136" y="123" width="58" height="34" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect>
				<rect x="194" y="123" width="58" height="34" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect>
				<text class="viz-axis-label" x="49" y="145" text-anchor="middle">0.04</text>
				<text class="viz-axis-label" x="107" y="145" text-anchor="middle">0.31</text>
				<text class="viz-axis-label" x="165" y="145" text-anchor="middle">0.08</text>
				<text class="viz-axis-label" x="223" y="145" text-anchor="middle">0.22</text>
				<text class="viz-callout" x="270" y="145" text-anchor="middle">LUT</text>
				<path d="M49 161V177M107 161V177M165 161V177M223 161V177" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<rect x="67" y="178" width="168" height="25" rx="12" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect>
				<text class="viz-callout" x="151" y="195" text-anchor="middle">approx. distance = 0.65</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> IVF first limits the search to the two centroid lists nearest Q, so the dashed lists are never scanned. PQ then scores each remaining candidate by summing one lookup per subvector code instead of reading and comparing every float. IVF reduces the candidate count; PQ reduces the cost per candidate.</figcaption>
</figure>

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
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity/).
- [RAG: retrieval-augmented generation](/concepts/rag-overview/).
