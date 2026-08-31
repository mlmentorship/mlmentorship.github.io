---
title: "Embedding spaces and similarity metrics"
description: "How learned vector representations encode meaning, and why cosine similarity is the default metric for retrieval and recsys."
date: "2026-02-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An embedding space is a learned vector space in which points represent objects (words, sentences, images, items, users) and geometric relationships. Distance, angle. Encode semantic relationships such as similarity or relevance.

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

<!-- visual:embedding-normalization-metric-equivalence -->
<figure class="learning-figure" aria-labelledby="embedding-normalization-title">
	<p class="visual-kicker">Metric geometry</p>
	<p class="visual-title" id="embedding-normalization-title">What changes when every embedding is projected onto the unit circle?</p>
	<div class="visual-grid--two">
		<section class="visual-panel plot-panel" aria-labelledby="raw-embedding-panel-title">
			<h4 id="raw-embedding-panel-title">Before: magnitude can win</h4>
			<p>Raw dot product rewards length as well as alignment.</p>
			<svg viewBox="0 0 300 260" role="img" aria-labelledby="raw-embedding-svg-title raw-embedding-svg-desc">
				<title id="raw-embedding-svg-title">Raw query and candidate embedding vectors</title>
				<desc id="raw-embedding-svg-desc">From a shared origin, unit query q points right. Candidate A is a longer solid ray ending in a circle at coordinates 1.8, 1.6. Candidate B is a shorter dashed ray ending in a diamond at 0.9, 0.2. Although B has the smaller angle to q, raw dot product ranks A first because A is longer.</desc>
				<rect class="viz-plot-bg" x="5" y="5" width="290" height="220" rx="3"></rect>
				<path class="viz-axis" d="M40 195 H278 M40 195 V18"></path>
				<text class="viz-label" x="28" y="210">0</text>
				<path class="viz-roc-curve" d="M40 195 L120 195"></path>
				<text class="viz-callout" x="94" y="214">q = (1, 0)</text>
				<path class="viz-pr-curve" d="M40 195 L184 67"></path>
				<circle class="viz-node" cx="184" cy="67" r="6"></circle>
				<text class="viz-callout" x="192" y="61">A = (1.8, 1.6)</text>
				<path class="viz-baseline" d="M40 195 L112 179"></path>
				<path class="viz-operating-point" d="M112 173 L118 179 L112 185 L106 179 Z"></path>
				<text class="viz-callout" x="121" y="176">B = (0.9, 0.2)</text>
				<text class="viz-callout" x="10" y="244">raw dot: A 1.80 &gt; B 0.90</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel" aria-labelledby="normalized-embedding-panel-title">
			<h4 id="normalized-embedding-panel-title">After: direction decides</h4>
			<p>Normalization preserves each angle and sets every length to 1.</p>
			<svg viewBox="0 0 300 260" role="img" aria-labelledby="normalized-embedding-svg-title normalized-embedding-svg-desc">
				<title id="normalized-embedding-svg-title">The query and candidates normalized onto a unit circle</title>
				<desc id="normalized-embedding-svg-desc">The same three vector directions now end on one circle. Candidate B's diamond endpoint is closer to query q than candidate A's circle endpoint. Dot product equals cosine, so both rank B first; Euclidean chord distance is also smaller for B.</desc>
				<rect class="viz-plot-bg" x="5" y="5" width="290" height="220" rx="3"></rect>
				<circle class="viz-gridline" cx="130" cy="125" r="90"></circle>
				<path class="viz-axis" d="M25 125 H280"></path>
				<path class="viz-roc-curve" d="M130 125 L220 125"></path>
				<text class="viz-callout" x="224" y="140">q</text>
				<path class="viz-pr-curve" d="M130 125 L197 65"></path>
				<circle class="viz-node" cx="197" cy="65" r="6"></circle>
				<text class="viz-callout" x="205" y="61">unit A</text>
				<path class="viz-baseline" d="M130 125 L218 105"></path>
				<path class="viz-operating-point" d="M218 99 L224 105 L218 111 L212 105 Z"></path>
				<text class="viz-callout" x="228" y="105">unit B</text>
				<path class="viz-operating-guide" d="M220 125 L197 65"></path>
				<path class="viz-operating-guide" d="M220 125 L218 105"></path>
				<text class="viz-label" x="46" y="27">all endpoints: length = 1</text>
				<text class="viz-callout" x="10" y="240">dot = cos: B 0.976 &gt; A 0.747</text>
				<text class="viz-callout" x="10" y="255">L2 chord: B 0.218 &lt; A 0.711</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> follow each ray from the origin. Before normalization, A's extra length overwhelms its worse angle, so dot product ranks A first. On the unit circle only direction remains: dot equals cosine, and the shorter L2 chord identifies the same winner, B.</figcaption>
</figure>

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
