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

**Learning objective:** trace which tower runs during a catalog refresh and which tower runs on every request, so you can explain why two-tower retrieval scales.

<!-- visual:two-tower-offline-online-split -->
<figure class="learning-figure plot-panel" aria-labelledby="two-tower-serving-heading">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="two-tower-serving-heading">Precompute the catalog side; run only the query side per request.</p>
	<svg viewBox="0 0 360 300" role="img" aria-labelledby="two-tower-serving-title two-tower-serving-desc">
		<title id="two-tower-serving-title">Offline item encoding feeds the index used by online query retrieval</title>
		<desc id="two-tower-serving-desc">The upper offline lane sends every catalog item through the item tower and writes the resulting item vectors to an approximate nearest-neighbor index during a batch refresh. The lower online lane sends one incoming query through the query tower to produce vector q. The index compares q with its stored item vectors and returns top-K candidate IDs. Text labels and a dashed batch-refresh path distinguish the two lifecycles without relying on color.</desc>
		<defs>
			<marker id="two-tower-solid-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" class="viz-arrow-forward"></path></marker>
			<marker id="two-tower-dashed-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" style="fill:var(--viz-state-stroke)"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="20" width="344" height="91" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="13">1 · OFFLINE OR WHEN THE CATALOG / MODEL CHANGES</text>
		<rect class="viz-node" x="18" y="42" width="88" height="40" rx="4"></rect>
		<text class="viz-label" x="62" y="58" text-anchor="middle">ALL CATALOG</text>
		<text class="viz-label" x="62" y="72" text-anchor="middle">ITEM FEATURES</text>
		<path d="M106 62H128" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#two-tower-solid-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="134" y="37" width="92" height="50" rx="4"></rect>
		<text class="viz-axis-label" x="180" y="57" text-anchor="middle">ITEM TOWER</text>
		<text class="viz-label" x="180" y="73" text-anchor="middle">run in batch</text>
		<path d="M226 62H244" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#two-tower-solid-arrow)"></path>
		<rect class="viz-node" x="250" y="42" width="92" height="40" rx="4" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-label" x="296" y="58" text-anchor="middle">ITEM VECTORS</text>
		<text class="viz-label" x="296" y="72" text-anchor="middle">i₁, i₂, …, iₙ</text>
		<text class="viz-callout" x="205" y="98" text-anchor="middle">batch refresh writes stored vectors</text>
		<rect class="viz-plot-bg" x="8" y="132" width="344" height="161" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="125">2 · ONLINE FOR EACH REQUEST</text>
		<path d="M296 82V101H111V248H126" fill="none" stroke="var(--viz-state-stroke)" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#two-tower-dashed-arrow)"></path>
		<rect class="viz-node" x="18" y="154" width="78" height="42" rx="4"></rect>
		<text class="viz-label" x="57" y="171" text-anchor="middle">ONE QUERY</text>
		<text class="viz-label" x="57" y="186" text-anchor="middle">+ context</text>
		<path d="M96 175H116" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#two-tower-solid-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="122" y="150" width="100" height="50" rx="4"></rect>
		<text class="viz-axis-label" x="172" y="170" text-anchor="middle">QUERY TOWER</text>
		<text class="viz-label" x="172" y="186" text-anchor="middle">run once</text>
		<path d="M222 175H244" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#two-tower-solid-arrow)"></path>
		<rect class="viz-node viz-node--input" x="250" y="154" width="92" height="42" rx="4"></rect>
		<text class="viz-label" x="296" y="171" text-anchor="middle">QUERY VECTOR</text>
		<text class="viz-label" x="296" y="186" text-anchor="middle">q ∈ Rᵈ</text>
		<path d="M296 196V214H180V219" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#two-tower-solid-arrow)"></path>
		<rect class="viz-node" x="132" y="225" width="96" height="48" rx="4" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-axis-label" x="180" y="244" text-anchor="middle">ANN INDEX</text>
		<text class="viz-label" x="180" y="260" text-anchor="middle">q · stored i</text>
		<path d="M228 249H246" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#two-tower-solid-arrow)"></path>
		<rect class="viz-node viz-node--output" x="252" y="228" width="90" height="42" rx="4"></rect>
		<text class="viz-axis-label" x="297" y="245" text-anchor="middle">TOP-K IDS</text>
		<text class="viz-label" x="297" y="260" text-anchor="middle">to ranker</text>
		<text class="viz-callout" x="18" y="287">No item-tower call occurs on this request path.</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the dashed refresh path once for the catalog: encode every item, then store its vector in the ANN index. For each request, follow only the solid lower path: encode one query, search the stored vectors, and send top-K IDs to the ranker. The speedup comes from moving item-tower work out of the request path. Original schematic checked against <a href="https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/">Yi et al.</a>, the <a href="https://www.tensorflow.org/recommenders/examples/basic_retrieval">TensorFlow Recommenders retrieval guide</a>, and <a href="https://cloud.google.com/architecture/implement-two-tower-retrieval-large-scale-candidate-generation">Google Cloud's two-tower serving architecture</a>.</figcaption>
</figure>

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
