---
title: "Two-tower retrieval"
description: "Separate query and catalog encoding, preserve candidate and model-version contracts, and measure approximate retrieval before reranking."
date: "2025-08-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A two-tower model independently encodes queries and items, then scores their vector interaction.
Independent item encoding permits reuse until its inputs or representation contract change.
Index approximation, candidate eligibility, negative semantics, and model-version compatibility remain separate correctness obligations.

## Architecture

```
query  → query_tower  → q ∈ R^d
item   → item_tower   → i ∈ R^d
score  = q · i  (or cosine)
```

The towers can share weights; normalization and vector dimension are design choices, not definitions.
For nonzero unit vectors, dot product equals cosine; otherwise magnitude changes dot-product ranking.
Stored item features must not require query-specific joint computation under this architecture.

<!-- visual:two-tower-offline-online-split -->
<figure class="learning-figure plot-panel" aria-labelledby="two-tower-serving-heading">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="two-tower-serving-heading">Encode items before the request; encode the query during it</p><div class="visual-argument" data-visual-argument="two-tower-offline-online-split" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="Item encoding happens before the request; query encoding happens during it. Offline item vectors populate the index. The live request encodes its query and searches that versioned index without re-encoding every item."><title>Item encoding happens before the request; query encoding happens during it</title><desc>Offline item vectors populate the index. The live request encodes its query and searches that versioned index without re-encoding every item.</desc><text x="147.835" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="147.835" dy="0">compute</tspan></text><text x="275.94500000000005" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="275.94500000000005" dy="0">index</tspan></text><text x="71.78" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">offline</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="83.78" y="36" width="123.11000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="145.335" y="71" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="145.335" dy="0">encode</tspan></text><path d="M88.78 89L201.89000000000001 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="211.89000000000001" y="36" width="123.11000000000001" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="273.44500000000005" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="273.44500000000005" dy="0">build</tspan></text></g><text x="71.78" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">online</tspan></text><g data-proof-status="value"><rect data-proof-cell="1,0" x="83.78" y="100" width="123.11000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="145.335" y="135" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="145.335" dy="0">query</tspan></text><path d="M88.78 153L201.89000000000001 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="211.89000000000001" y="100" width="123.11000000000001" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="273.44500000000005" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="273.44500000000005" dy="0">search</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">the dashed refresh path computes and stores catalog vectors. The request path computes a query and searches those compatible stored vectors. The original schematic follows <a href="https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/">Yi et al.</a>, <a href="https://www.tensorflow.org/recommenders/examples/basic_retrieval">TensorFlow Recommenders</a>, and <a href="https://cloud.google.com/architecture/implement-two-tower-retrieval-large-scale-candidate-generation">Google Cloud</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="two-tower-offline-online-split"><strong>Check:</strong> Which computation was already done before this request? What version change could require rebuilding its stored item vectors?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="two-tower-offline-online-split">Offline item vectors populate the index. The live request encodes its query and searches that versioned index without re-encoding every item.</span></figcaption>
</figure>

## Training

With query matrix $Q\in\mathbb R^{B\times d}$ and item matrix $I\in\mathbb R^{M\times d}$, scores are $QI^T/\tau$.
Require positive temperature and declare acceptable positives and eligible candidates per row.

### In-batch sampled softmax
The one-positive, temperature-one case uses:

$$
L = -\log \frac{\exp(q \cdot i^+)}{\sum_{j=1}^{B} \exp(q \cdot i_j)}.
$$

For $q=(1,0)$, positive A $=(1,0)$, and valid negative B $=(0,1)$, scores are one and zero.
Positive probability is $e/(e+1)\approx0.731059$ and loss is $\log(1+e^{-1})\approx0.313262$.
If B is also acceptable, treating it as a known negative is wrong.

Deduplicate item IDs first; this implementation maximizes total probability of the acceptable-positive set, not each positive's individual probability.
It uses Boolean masks and float64 scores with representable score gaps:

```python
import numpy as np

def retrieval_loss(query, items, positive_mask, valid_mask, temperature=1.0):
	query = np.asarray(query, dtype=np.float64)
	items = np.asarray(items, dtype=np.float64)
	positive = np.asarray(positive_mask)
	valid = np.asarray(valid_mask)
	if query.ndim != 2 or items.ndim != 2:
		raise ValueError("Expected two embedding matrices")
	if min(query.shape) == 0 or min(items.shape) == 0:
		raise ValueError("Embedding axes must be nonempty")
	if query.shape[1] != items.shape[1]:
		raise ValueError("Embedding dimensions differ")
	shape = (len(query), len(items))
	if positive.shape != shape or valid.shape != shape:
		raise ValueError("Mask shape differs from score shape")
	if positive.dtype != np.bool_ or valid.dtype != np.bool_:
		raise ValueError("Masks must be Boolean")
	if np.any(positive & ~valid) or not np.all(positive.any(axis=1)):
		raise ValueError("Each row needs eligible positive candidates")
	if not np.isfinite(temperature) or temperature <= 0:
		raise ValueError("Temperature must be finite and positive")
	if not np.isfinite(query).all() or not np.isfinite(items).all():
		raise ValueError("Embeddings must be finite")
	with np.errstate(over="raise", invalid="raise", divide="raise"):
		scores = query @ items.T / temperature
		scores = np.where(valid, scores, -np.inf)
		relative = scores - scores.max(axis=1, keepdims=True)
		denominator = np.logaddexp.reduce(relative, axis=1)
		numerator = np.logaddexp.reduce(
			np.where(positive, relative, -np.inf), axis=1
		)
		return float(np.mean(denominator - numerator))
```

Both candidates acceptable gives loss zero; rejecting every positive raises instead of silently creating an undefined objective.

### Need Joint Token Detail

**New task:** Distinguish subtle query-item phrase relations lost by independent compression.

**Picture and change:** Evaluate a bounded cross-encoder or late-interaction reranker with its extra work and storage.

**Check:** A reranker cannot recover an omitted relevant item, so inspect candidate recall separately.

**Use when:** Added pair detail improves the measured decision under its budget.

### Change the Item Model

**New task:** Retrain the item encoder or tokenizer.

**Picture and change:** Re-encode/reindex or prove a compatible migration; version the query tower, analyzer, scoring policy, and index together.

**Check:** Rotating only the query from $(1,0)$ to $(0,1)$ reverses A/B scores against the unchanged index.
Equal vector dimension is insufficient evidence of compatibility.

**Use when:** Stored representation semantics change; retain an atomic rollback bundle and enforce current eligibility/deletions after retrieval.

<!-- learning:core-end -->
<div data-learning-extension>

### Importance-corrected sampled softmax
Sampled-softmax correction must specify proposal probabilities, inclusion multiplicities, positive handling, and the intended full-catalog objective.
Subtracting a log-proposal term is not a universal exposure-bias cure; follow the estimator's derivation, such as [Yi et al.](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/).

### Hard negative mining
High-scoring alternatives may be unlabeled positives; validate their negative status and evaluate the changed sampling distribution.

## Two-stage architecture

Candidate generation followed by richer ranking is one useful arrangement, not a required architecture or guaranteed latency target.
Approximate search trades recall, memory, and work under its actual index and workload; sublinear scaling is not unconditional.

## Tradeoffs vs. cross-encoder

| Property | Two-tower | Cross-encoder |
|---------|-----------|---------------|
| Pair computation | Independent encodings plus vector interaction | Joint query/item computation |
| Reusable catalog state | Stored item vectors | Architecture-dependent; joint scores still need the query |
| Quality/cost | Measure for task and budget | Measure for task and budget |

## Common pitfalls

- **No interactions:** dot product is an interaction, but not arbitrary joint token computation.
- **Incompatible updates:** changing only the query tower can also invalidate the index.
- **Unproven quality:** reranking and negative mining require evaluation, not an assumed improvement.

## Related

- [Embedding spaces](/concepts/embedding-spaces-and-similarity/). Vector representations and indexing.
- [RAG overview](/concepts/rag-overview/). Retrieval-augmented generation uses two-tower for the retrieval step.

</div>
