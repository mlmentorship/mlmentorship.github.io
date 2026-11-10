---
title: "Embedding spaces and similarity metrics"
description: "Choose dot product, cosine, or distance from the retrieval objective, then verify normalization and approximate indexing."
date: "2026-02-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An embedding maps an object to a vector; its training objective determines which vector relationships should predict relevance.
Choose whether length, direction, or absolute distance should affect retrieval before choosing the index.
Normalization changes the ranking objective unless the model and task already make length irrelevant.

## Common similarity metrics

For two vectors $u, v \in \mathbb{R}^d$:

| Metric | Formula | When to use |
|--------|---------|-------------|
| **Dot product** | $u^\top v$ | When magnitudes are meaningful (e.g., learned matrix-factorization scores). Indexable with MIPS algorithms. |
| **Cosine** | $\frac{u^\top v}{\|u\|\,\|v\|}$ | Default for embeddings where direction encodes meaning and magnitude is a noise/popularity confound. |
| **Euclidean / L2** | $\|u - v\|_2$ | When distances have physical meaning (image patches in pixel space, geographic coordinates). |
| **Negative Euclidean²** | $-\|u-v\|^2$ | Equivalent to dot product on L2-normalized vectors plus a constant. |

Cosine requires nonzero vectors and explicitly discards positive scale.
Some models train with normalized similarities; others use unnormalized scores.
Follow the model's documented scoring contract instead of assuming all embedding models share one metric.

## L2 normalization

For nonzero $u,v$, define $\widehat u=u/\lVert u\rVert$ and $\widehat v=v/\lVert v\rVert$.
Then

$$
\widehat u^\top\widehat v=\cos(u,v),\qquad
\lVert\widehat u-\widehat v\rVert^2=2-2\cos(u,v).
$$

Thus normalized dot product, cosine, and normalized squared distance induce the same exact ranking, including ties.
The identity follows by expanding the squared difference and substituting both unit norms.

Take query $q=(1,0)$ and candidates $A=(1.8,1.6)$, $B=(0.9,0.2)$.
Raw dot product favors A: $1.8>0.9$.
Cosine favors B: approximately $0.976>0.747$.
Their normalized chord distances are approximately $0.218<0.711$.

<!-- visual:embedding-normalization-metric-equivalence -->
<figure class="learning-figure" aria-labelledby="embedding-normalization-title">
	<p class="visual-kicker">Metric geometry</p>
	<p class="visual-title" id="embedding-normalization-title">Length can win a raw match; normalization compares directions on the unit circle</p><div class="visual-argument" data-visual-argument="embedding-normalization-metric-equivalence" data-argument-kind="bars" data-argument-composition="raw-and-normalized-vectors"><svg viewBox="0 0 360 1220" role="img" aria-label="Length can win a raw match; normalization compares directions on the unit circle. Raw A has a larger projection along the query because it is long. Normalization preserves each direction but puts both candidates on the unit circle. Aligned B then wins; its point coincides with the query. On this circle, greater projection also means shorter query distance. The normalized chart uses a fivefold zoom; circle radius remains one. Lengthening A alone cannot move its normalized point."><title>Length can win a raw match; normalization compares directions on the unit circle</title><desc>Raw A has a larger projection along the query because it is long. Normalization preserves each direction but puts both candidates on the unit circle. Aligned B then wins; its point coincides with the query. On this circle, greater projection also means shorter query distance. The normalized chart uses a fivefold zoom; circle radius remains one. Lengthening A alone cannot move its normalized point.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Raw view: preserve query, A and B</tspan><tspan x="20" dy="18">identities</tspan></text><path d="M35 310L332 310" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M60 68L60 337" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="60" cy="310" r="20" data-picture-unit-circle="0" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.5;stroke-dasharray:4 4"></circle><path d="M60 310L60 316" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="60" y="353" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="60" dy="0">0</tspan></text><path d="M160 310L160 316" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="160" y="353" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="160" dy="0">5</tspan></text><path d="M260 310L260 316" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="260" y="353" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="260" dy="0">10</tspan></text><g data-picture-embedding-member="0,0" data-picture-vector="10,10"><path d="M60 310L260 110M257.17157287525384 118.48528137423857L260 110L251.51471862576145 112.8284271247462" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="260" cy="110" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M260 110L260 310" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="290" y="92" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="290" dy="0">A</tspan></text><g data-picture-embedding-member="0,1" data-picture-vector="1,0"><path d="M60 310L80 310M72 314L80 310L72 306" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="80" cy="310" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><circle cx="80" cy="310" r="9" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><circle cx="80" cy="310" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="274" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Query and B</tspan></text><text x="20" y="384" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Query: hollow ring. B: inner dot at</tspan><tspan x="20" dy="18">the same point.</tspan></text><text x="20" y="445" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Raw query projection: A 10; B 1. A</tspan><tspan x="20" dy="18">wins.</tspan></text><text x="20" y="547" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Normalize both; enlarge the unit</tspan><tspan x="20" dy="18">circle five times</tspan></text><path d="M35 830L332 830" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M145 588L145 954" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="145" cy="830" r="100" data-picture-unit-circle="1" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.5;stroke-dasharray:4 4"></circle><text x="245" y="870" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="245" dy="0">1</tspan></text><text x="130" y="729" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="130" dy="0">1</tspan></text><g data-picture-embedding-member="1,0" data-picture-vector="0.7071067811865475,0.7071067811865475"><path d="M145 830L215.71067811865476 759.2893218813452M212.88225099390857 767.7746032555839L215.71067811865476 759.2893218813452L207.2253967444162 762.1177490060915" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="215.71067811865476" cy="759.2893218813452" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M215.71067811865476 759.2893218813452L215.71067811865476 830" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="215.71067811865476" y="741.2893218813452" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="215.71067811865476" dy="0">A</tspan></text><g data-picture-embedding-member="1,1" data-picture-vector="1,0"><path d="M145 830L245 830M237 834L245 830L237 826" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="245" cy="830" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><circle cx="245" cy="830" r="9" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><circle cx="245" cy="830" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="300" y="794" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="300" dy="0">Query and</tspan><tspan x="300" dy="18">B</tspan></text><text x="20" y="991" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Query: hollow ring. B: inner dot at</tspan><tspan x="20" dy="18">the same point.</tspan></text><text x="20" y="1050" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Query projection: A 0.7071; B 1. B</tspan><tspan x="20" dy="18">wins.</tspan></text><path d="M215.71067811865476 759.2893218813452L245 830" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="1119" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Query distance: A 0.7654; B zero.</tspan><tspan x="20" dy="18">Longer raw A stays at this same</tspan><tspan x="20" dy="18">normalized point.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> raw dot product favors the longer A. Unit normalization preserves angles but discards length. Cosine, normalized dot product, and normalized chord distance then agree on B.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="embedding-normalization-metric-equivalence"><strong>Check:</strong> Why does B win both normalized dot and nearest L2 even though A wins the raw dot product?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="embedding-normalization-metric-equivalence">Raw A has a larger projection along the query because it is long. Normalization preserves each direction but puts both candidates on the unit circle. Aligned B then wins; its point coincides with the query. On this circle, greater projection also means shorter query distance. The normalized chart uses a fivefold zoom; circle radius remains one. Lengthening A alone cannot move its normalized point. <strong>After normalization, can increasing only A's original length improve its cosine?</strong></span></figcaption>
</figure>

Normalize stored candidates once when cosine is the intended objective.
Normalize each nonzero query or divide scores by its norm when actual cosine values are needed.
Declare a rejection or fallback policy for zero vectors; adding an arbitrary epsilon changes the score definition.

## Change the Requirement

### Retain Meaningful Magnitude

**New task:** Serve a model trained to encode relevance through both vector length and direction.

**Picture and change:** Use the raw dot product rather than projecting onto the unit sphere.

**Check:** The same fixture now ranks A above B, matching its unnormalized objective.

**Use when:** Training and evaluation support that scoring contract.
Length may be useful signal or an unwanted association; geometry alone cannot determine which.

### Trade Exact Search for Candidate Speed

**New task:** Retrieve from an approximate index under a latency budget.

**Picture and change:** Candidate generation may omit the exact nearest item; reranking cannot recover an absent candidate.

**Check:** If exact top-one is B but the index returns only A, top-one recall is zero despite correct reranking.

**Use when:** Measured latency and recall justify approximation under the same metric, filters, and data version.

<!-- learning:core-end -->
<div data-learning-extension>

## Geometry of learned embeddings

Clusters, linear analogies, anisotropy, and distance concentration depend on the model and input distribution.
Whitening or centering changes geometry; validate relevance afterward rather than assuming improvement.
Compare dimensions through measured retrieval quality, memory, and latency, not a universal preferred range.

## Indexing for fast retrieval

For $N$ items and queries:

| Method | Decision to measure |
| --- | --- |
| Exact scan | $O(Nd)$ arithmetic; establishes the same-metric reference ranking |
| [HNSW](https://arxiv.org/abs/1603.09320) | Graph-search breadth, build cost, recall, and memory |
| IVF with product quantization | Number of searched partitions and quantization error |
| ScaNN | Candidate-search and quantization settings under the workload |

Approximate latency and recall are empirical workload properties, not universal logarithmic guarantees or fixed percentages.

## Common pitfalls

- **Mixed objectives.** Normalizing only some candidates invalidates the unit-vector equivalence.
- **Mixed coordinate systems.** Independently trained encoders need a validated shared-space or alignment contract.
- **Unproved bias claims.** Removing length changes scores; it does not establish fairer or better retrieval.

## Reconstruct

<details>
<summary>Can exact reranking fix a missing relevant candidate?</summary>

No. Reranking only reorders supplied candidates.
Measure candidate coverage separately from ranking quality, including filtered and tail-query populations.

</details>

</div>
