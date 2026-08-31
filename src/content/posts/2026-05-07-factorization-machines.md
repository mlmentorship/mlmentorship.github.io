---
title: "Factorization machines"
description: "Linear models can't capture feature interactions. Polynomial models have too many parameters. Factorization machines find a middle path: factorize the interaction matrix and learn an embedding per feature."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **factorization machine** ([Rendle, 2010](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf)) models pairwise feature interactions as $\langle v_i, v_j \rangle$ where each feature $i$ has an embedding $v_i \in \mathbb{R}^k$. The full prediction is

$$
\hat{y}(x) = w_0 + \sum_i w_i x_i + \sum_{i < j} \langle v_i, v_j \rangle x_i x_j.
$$

Linear models (logistic regression) are fast but miss interactions. A degree-2 polynomial model has $\binom{d}{2}$ interaction parameters, which is infeasible at $d = 10^6$ (typical for sparse categorical features) and learns nothing for unseen pairs. FMs sidestep both problems by factorizing the interaction matrix into rank-$k$ embeddings, sharing parameters across pairs.

Result: the FM has $O(d k)$ parameters instead of $O(d^2)$, and it generalizes to unseen feature pairs because it only needs to have seen each feature, not each pair. This made FMs the default tabular-recsys model from roughly 2010 to 2018, and they remain a strong baseline today.

<!-- visual:fm-shared-embeddings-unseen-pair -->
<figure class="learning-figure plot-panel" aria-labelledby="fm-sharing-visual-title">
	<p class="visual-kicker">Parameter sharing</p>
	<p class="visual-title" id="fm-sharing-visual-title">An unseen pair has no pair-specific weight, but it still has a factorized coefficient.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 430" role="img" aria-labelledby="fm-sharing-svg-title fm-sharing-svg-desc">
			<title id="fm-sharing-svg-title">Pair-specific weights compared with shared feature embeddings for an unseen interaction</title>
			<desc id="fm-sharing-svg-desc">An original three-feature example compares two parameterizations. In a polynomial model, training included pairs A-B and B-C, so their separate edge weights are learned, while the never-observed A-C edge has an unknown pair-specific weight. In a factorization machine, each feature instead owns one reusable two-dimensional vector: A is one-zero, B is point-eight-point-two, and C is point-six-point-four. The observed A-B and B-C pairs help learn those vectors. Even though A-C never appeared together, its coefficient is defined and computable as the dot product of A and C, one times point six plus zero times point four, which equals point six. This demonstrates availability of an estimate, not guaranteed accuracy on every unseen pair.</desc>
			<text class="viz-axis-label" x="14" y="20">PAIR-SPECIFIC MODEL · EDGE PARAMETERS</text>
			<rect class="viz-node viz-node--input" x="18" y="42" width="76" height="38" rx="4"></rect>
			<text class="viz-callout" x="56" y="65" text-anchor="middle">feature A</text>
			<rect class="viz-node viz-node--input" x="142" y="42" width="76" height="38" rx="4"></rect>
			<text class="viz-callout" x="180" y="65" text-anchor="middle">feature B</text>
			<rect class="viz-node viz-node--input" x="266" y="42" width="76" height="38" rx="4"></rect>
			<text class="viz-callout" x="304" y="65" text-anchor="middle">feature C</text>
			<path class="viz-pr-curve" d="M94 61H142 M218 61H266"></path>
			<text class="viz-label" x="118" y="51" text-anchor="middle">wAB</text>
			<text class="viz-label" x="242" y="51" text-anchor="middle">wBC</text>
			<path class="viz-baseline" d="M56 84C56 132 304 132 304 84"></path>
			<rect class="viz-node viz-node--focus" x="107" y="102" width="146" height="42" rx="4"></rect>
			<text class="viz-callout" x="180" y="120" text-anchor="middle">wAC = ?</text>
			<text class="viz-label" x="180" y="136" text-anchor="middle">A–C never appeared together</text>
			<path class="viz-gridline" d="M8 164H352"></path>
			<text class="viz-axis-label" x="14" y="188">FACTORIZATION MACHINE · PARAMETERS LIVE ON FEATURES</text>
			<rect class="viz-node viz-node--input" x="18" y="207" width="96" height="62" rx="4"></rect>
			<text class="viz-callout" x="66" y="228" text-anchor="middle">feature A</text>
			<text class="viz-label" x="66" y="248" text-anchor="middle">vA = [1, 0]</text>
			<text class="viz-label" x="66" y="262" text-anchor="middle">learned via A–B</text>
			<rect class="viz-node viz-node--input" x="132" y="207" width="96" height="62" rx="4"></rect>
			<text class="viz-callout" x="180" y="228" text-anchor="middle">feature B</text>
			<text class="viz-label" x="180" y="248" text-anchor="middle">vB = [.8, .2]</text>
			<text class="viz-label" x="180" y="262" text-anchor="middle">shared by both</text>
			<rect class="viz-node viz-node--input" x="246" y="207" width="96" height="62" rx="4"></rect>
			<text class="viz-callout" x="294" y="228" text-anchor="middle">feature C</text>
			<text class="viz-label" x="294" y="248" text-anchor="middle">vC = [.6, .4]</text>
			<text class="viz-label" x="294" y="262" text-anchor="middle">learned via B–C</text>
			<path class="viz-pr-curve" d="M114 226H132 M228 226H246"></path>
			<text class="viz-label" x="123" y="216" text-anchor="middle">AB</text>
			<text class="viz-label" x="237" y="216" text-anchor="middle">BC</text>
			<path class="viz-baseline" d="M66 274C66 290 72 298 85 302 M275 302C288 298 294 290 294 274"></path>
			<text class="viz-callout" x="180" y="309" text-anchor="middle">unseen pair · coefficient still defined</text>
			<rect class="viz-node viz-node--focus" x="32" y="326" width="296" height="72" rx="4"></rect>
			<text class="viz-callout" x="180" y="348" text-anchor="middle">wAC := ⟨vA, vC⟩</text>
			<text class="viz-callout" x="180" y="369" text-anchor="middle">= 1 × .6 + 0 × .4 = .60</text>
			<text class="viz-label" x="180" y="388" text-anchor="middle">an available estimate, not a guarantee of accuracy</text>
			<text class="viz-axis-label" x="180" y="424" text-anchor="middle">O(d²) EDGE WEIGHTS → O(dk) FEATURE EMBEDDINGS</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> the polynomial model needs a separately learned parameter on every edge, so its unseen A–C edge has no estimate. The FM learns one vector per feature through observed pairs; once A and C each have a vector, their dot product defines the A–C coefficient even though that pair was absent. Original example based on <a href="https://doi.org/10.1109/ICDM.2010.127">Rendle (2010)</a>.</figcaption>
</figure>

## The mechanism

Each feature $i$ gets a weight $w_i \in \mathbb{R}$ (linear term) and an embedding $v_i \in \mathbb{R}^k$ (interaction term). The prediction includes:

- A global bias $w_0$.
- Per-feature linear terms $\sum_i w_i x_i$.
- Pairwise interactions $\sum_{i < j} \langle v_i, v_j \rangle x_i x_j$.

The naive interaction sum is $O(d^2)$ to evaluate, but Rendle showed it can be reformulated as

$$
\sum_{i < j} \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^{k} \left( \left(\sum_i v_{i,f} x_i\right)^2 - \sum_i v_{i,f}^2 x_i^2 \right).
$$

Linear in $d$. This is the trick that makes FMs scalable.

## Sparse one-hot inputs

The natural use case: categorical features, one-hot encoded. Each user-id, item-id, or category becomes a feature $i$ with embedding $v_i$. The pairwise interaction $\langle v_i, v_j \rangle x_i x_j$ is nonzero only when both $x_i = x_j = 1$, i.e. only between active feature pairs.

For a (user, item) example with one-hot features, the prediction is:

$$
\hat{y} = w_0 + w_{\text{user}} + w_{\text{item}} + \langle v_{\text{user}}, v_{\text{item}} \rangle.
$$

This is exactly a matrix factorization recsys model with bias terms. FMs **generalize** matrix factorization to arbitrary numbers of features (user, item, category, time, device), all sharing the same embedding mechanism.

## Variants

- **Field-aware FM (FFM)** ([Juan et al., 2016](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)). Each feature has multiple embeddings, one per "field" (e.g. user-feature embeddings paired against item-features differ from user-feature embeddings paired against time-of-day). More parameters, better accuracy on click prediction.
- **DeepFM** ([Guo et al., 2017](https://arxiv.org/abs/1703.04247)). Add a deep MLP on top of the same embeddings to capture high-order interactions. The dominant CTR-prediction architecture in industry from 2017 onwards.
- **xDeepFM**, **AutoInt**, **DCN**: subsequent variations layering self-attention or explicit cross-feature networks over the FM embedding base.

## Tradeoffs

| | |
|---|---|
| **vs logistic regression** | Captures pairwise interactions; needs more compute and tuning |
| **vs polynomial regression** | $O(d k)$ vs $O(d^2)$ parameters; generalizes to unseen pairs |
| **vs deep learning on raw features** | FM is simpler, trains faster, more interpretable; deep nets can capture higher-order interactions |
| **vs matrix factorization** | FM generalizes MF to many sparse features beyond just (user, item) |

For tabular click-through-rate prediction with high-cardinality categoricals, an FM-style embedding base (FM, DeepFM, FFM) is still the right starting point.

## Common pitfalls

- **Choosing $k$ too large**. $k = 8$ to $32$ is typical; larger $k$ overfits and is slower.
- **Forgetting the linear term**. The pairwise interactions cannot model main effects; both terms matter.
- **Using FM on dense numeric features without binning**. Dense features can be used, but the interaction $\langle v_i, v_j \rangle x_i x_j$ scales with the products, and the model is more sensitive to feature scaling. Bin or normalize first.
- **Ignoring regularization**. L2 on $v$ is essential when most features are rare.
- **Comparing FM to LR without matched features.** FM benefits from rich categorical features; on a clean numeric baseline it often loses to LR or gradient boosting.

## Related

- [Matrix factorization for recsys](/concepts/matrix-factorization-recsys/).
- [Alternating least squares](/concepts/alternating-least-squares/).
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity/).
