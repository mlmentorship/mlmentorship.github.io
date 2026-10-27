---
title: "Approximate nearest neighbors: HNSW, IVF, and product quantization"
description: "Separate candidate routing from vector compression, then measure neighbor recall, task quality, latency and complete index memory against exact search."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

Approximate nearest-neighbor indexes save routing work or vector bytes, creating different errors that need separate checks.

## Summary

Which neighbors disappear because routing skipped them, and which change order because distances were approximated?
First define the eligible catalog, metric, query distribution, $K$ and deterministic tie policy.
Exact top-$K$ under that same contract is the neighbor-recall reference, not a relevance oracle.

## The three families

### HNSW (Hierarchical Navigable Small World)

Route through sparse upper graph layers, then search a broader bottom-layer frontier.
Neighbor selection is heuristic, not an exact nearest-neighbor guarantee.
Record degree parameter $M$, construction breadth and query breadth `efSearch`.

### IVF (Inverted File Index)

Assign stored vectors to coarse centroids, then scan selected inverted lists.
`nlist` controls partitioning; `nprobe` controls lists visited.
A nearby point can belong to a skipped list.

### PQ (Product Quantization)

Split a vector into $m$ subvectors and store one $b$-bit code per subvector.
For squared Euclidean search, build query-to-codeword distance tables and sum selected entries.
The original query need not be quantized.
A 1024-dimensional float32 payload uses 4096 bytes; sixty-four eight-bit codes use 64 bytes.

### OPQ (Optimized PQ)

Learn a rotation before PQ to redistribute quantization difficulty across subvectors.
Measure improvement; it is not guaranteed for every dataset, query or trained codebook.

## Production combinations

IVF-PQ combines routing and compression.
Graph routing can also use compressed distances or supply candidates for exact reranking.
An IVF-HNSW index may use a graph over coarse centroids; inspect its actual implementation roles.

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
	<figcaption><strong>Read it this way:</strong> IVF selects lists; dashed lists are skipped. PQ sums the four selected lookup entries to 0.65 for each retained candidate. Candidate omission and distance distortion are separate failure paths.</figcaption>
</figure>

The lookup sum is $0.04+0.31+0.08+0.22=0.65$.
Now compare returned IDs with exact neighbors:

| Set | IDs |
| --- | --- |
| Exact top-five | A, B, C, D, E |
| ANN top-five | A, C, D, F, G |
| Retained exact IDs | A, C, D |
| Missing exact IDs | B, E |

Neighbor recall@5 is $3/5=0.6$.
Exact reranking cannot recover B or E from this fixed set.

## Change the resource constraint

### Increase search breadth

**New task:** raise neighbor recall above this floor.

**Picture and change:** increase `efSearch` or `nprobe`, holding queries, filters and catalog fixed.

**Check:** if broader routing adds B but not E, exact reranking can achieve $4/5=0.8$ here.
It still cannot produce the missing E.
Record candidate recall and tail latency, not only reranking accuracy.

**Use when:** omission, rather than retained-candidate ordering, dominates.

### Reduce stored bytes

**New task:** one million raw vectors exceed the memory budget.

**Picture and change:** raw payload is 4.096 decimal GB; the stated PQ codes use 64 MB.

**Check:** eight-byte IDs add another 8 MB before codebooks, lists, graph links and allocator overhead.
Keeping raw vectors for reranking retains their storage cost somewhere.
The 64-fold payload reduction is not a whole-service memory or speed guarantee.

**Use when:** storage or vector transfer is binding; evaluate compression, encoder dimension or tiering with quality controls.

## Reconstruct the error

Could neighbor recall decrease without changing useful-document recall?

<details>
<summary>Reasoned answer</summary>

Yes, if omitted neighbors are irrelevant or redundant.
Alternatively one missing ID can remove the only useful item.
Measure neighbor approximation and task relevance separately.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Recall vs latency vs memory

Measure a workload-specific frontier, not a universal pick-two rule.
Include query concurrency, filters, vector distribution, build/update cost and resident plus peak memory.
Graph links, IDs, codebooks, deletions and optional originals all contribute.
Exact search has finite $O(Nd)$ work and can be preferable for small or highly filtered candidate sets.
For dot products, state whether a multiply-add counts as one operation or two FLOPs.
Bandwidth may dominate arithmetic.
ANN does not guarantee sublinear worst-case work.

## Common pitfalls

Report $K$, ground-truth metric, ties and filtered eligibility with recall.
Normalize vectors consistently when cosine is implemented through inner product.
Check versioned insertion, deletion, tombstone, retraining and filter behavior rather than inferring it from a family name.
Strong static unfiltered recall does not qualify a frequently updated filtered service.
Original algorithms: <a href="https://arxiv.org/abs/1603.09320">HNSW</a> and <a href="https://hal.inria.fr/inria-00514462/document">product quantization</a>.

## Related

- [Two-tower retrieval](/concepts/two-tower-retrieval/).
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity/).
- [RAG: retrieval-augmented generation](/concepts/rag-overview/).

</div>
