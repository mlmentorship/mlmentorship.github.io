---
title: "Matrix factorization for recsys"
description: "Decompose the user-item interaction matrix into user and item embeddings whose dot product approximates the rating. The original collaborative filtering."
date: "2026-02-16"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Matrix factorization** for collaborative filtering [(Koren et al., 2009)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) factorizes a sparse user-item rating matrix $R \in \mathbb{R}^{m \times n}$ into two low-rank matrices: $R \approx U V^\top$ where $U \in \mathbb{R}^{m \times k}$ contains user embeddings and $V \in \mathbb{R}^{n \times k}$ contains item embeddings. Predicted rating: $\hat r_{ui} = u_u^\top v_i$.

MF was the dominant collaborative-filtering method from the Netflix Prize era (2006–2009) through about 2018. It still underlies modern two-tower retrieval, embedding-based recsys, and has clean equivalences to many later techniques (matrix completion, neural MF, etc.). Knowing MF makes the move to two-tower neural models obvious.

<!-- visual:mf-missing-rating-dot-product -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="mf-rating-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="mf-rating-visual-title">Trace one missing rating from its matrix indices to a dot-product prediction.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 360" role="img" aria-labelledby="mf-rating-svg-title mf-rating-svg-desc">
			<title id="mf-rating-svg-title">A missing rating predicted from one user vector and one item vector</title>
			<desc id="mf-rating-svg-desc">In a sparse ratings matrix, the missing entry in user A's row and item 3's column is selected. Row A of the user-factor matrix gives vector u A equal to 2 comma 1. Row 3 of the item-factor matrix gives vector v 3 equal to 1.5 comma 1. Matching latent-factor coordinates are multiplied and summed: 2 times 1.5 plus 1 times 1 equals a predicted rating of 4.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="24" y="22">1 · LOCATE THE MISSING ENTRY</text>
			<rect class="viz-node viz-node--input" x="30" y="42" width="205" height="250" rx="10"></rect>
			<text class="viz-node-label" x="132" y="67">Sparse ratings R</text>
			<text class="viz-node-value" x="132" y="85">columns are items</text>
			<text class="viz-label" x="91" y="109">1</text>
			<text class="viz-label" x="126" y="109">2</text>
			<text class="viz-callout" x="161" y="109">3</text>
			<text class="viz-label" x="196" y="109">4</text>
			<text class="viz-callout" x="50" y="140">A</text>
			<text class="viz-label" x="50" y="180">B</text>
			<text class="viz-label" x="50" y="220">C</text>
			<text class="viz-label" x="50" y="260">D</text>
			<path class="viz-gridline" d="M72 116 H212 M72 156 H212 M72 196 H212 M72 236 H212 M72 276 H212 M72 116 V276 M107 116 V276 M142 116 V276 M177 116 V276 M212 116 V276"></path>
			<text class="viz-node-value" x="89" y="141">5</text>
			<text class="viz-node-value" x="124" y="141">3</text>
			<rect class="viz-node viz-node--focus" x="143" y="117" width="33" height="38" rx="3"></rect>
			<text class="viz-node-label" x="159" y="142">?</text>
			<text class="viz-node-value" x="194" y="141">1</text>
			<text class="viz-node-value" x="89" y="181">4</text>
			<text class="viz-node-value" x="159" y="181">2</text>
			<text class="viz-node-value" x="124" y="221">1</text>
			<text class="viz-node-value" x="194" y="221">5</text>
			<text class="viz-node-value" x="89" y="261">2</text>
			<text class="viz-node-value" x="159" y="261">4</text>
			<text class="viz-edge-label" x="160" y="313">target cell R_A3</text>
			<path class="viz-forward" d="M236 136 H271"></path>
			<text class="viz-axis-label" x="280" y="22">2 · SELECT MATCHING FACTOR ROWS</text>
			<rect class="viz-node viz-node--focus" x="280" y="68" width="185" height="90" rx="10"></rect>
			<text class="viz-node-label" x="372" y="94">user row u_A</text>
			<text class="viz-node-value" x="372" y="118">from U ∈ R^(m×k)</text>
			<text class="viz-node-label" x="372" y="143">[ 2 , 1 ]</text>
			<rect class="viz-node viz-node--output" x="280" y="196" width="185" height="90" rx="10"></rect>
			<text class="viz-node-label" x="372" y="222">item row v_3</text>
			<text class="viz-node-value" x="372" y="246">from V ∈ R^(n×k)</text>
			<text class="viz-node-label" x="372" y="271">[ 1.5 , 1 ]</text>
			<path class="viz-forward" d="M466 113 C492 113 492 132 510 132"></path>
			<path class="viz-forward" d="M466 241 C492 241 492 220 510 220"></path>
			<text class="viz-axis-label" x="520" y="22">3 · MULTIPLY COORDINATES, THEN SUM</text>
			<rect class="viz-node" x="520" y="68" width="210" height="218" rx="10"></rect>
			<text class="viz-node-value" x="625" y="101">latent factor 1</text>
			<text class="viz-node-label" x="625" y="126">2 × 1.5 = 3</text>
			<text class="viz-node-value" x="625" y="159">latent factor 2</text>
			<text class="viz-node-label" x="625" y="184">1 × 1 = 1</text>
			<path class="viz-gridline" d="M548 203 H702"></path>
			<text class="viz-node-value" x="625" y="228">dot product: 3 + 1</text>
			<rect class="viz-node viz-node--focus" x="561" y="242" width="128" height="31" rx="6"></rect>
			<text class="viz-node-label" x="625" y="264">r̂_A3 = 4</text>
			<text class="viz-edge-label" x="380" y="329">R_A3 ≈ u_A^T v_3: the shared k coordinates are learned from observed ratings.</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> the missing cell supplies two indices: user A selects row A of <var>U</var>, and item 3 selects row 3 of <var>V</var>. Multiply matching latent coordinates and add them: 2 × 1.5 + 1 × 1 = 4, the prediction placed at <var>R</var><sub>A3</sub>.</figcaption>
</figure>

## The objective

Minimize regularized squared error on observed ratings $\Omega = \{(u, i) : r_{ui} \text{ observed}\}$:

$$
\min_{U, V} \sum_{(u, i) \in \Omega} (r_{ui} - u_u^\top v_i)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2).
$$

Often add bias terms: $\hat r_{ui} = \mu + b_u + b_i + u_u^\top v_i$ ($\mu$ = global mean, $b_u, b_i$ = user/item biases).

## Training

Loss is **non-convex jointly in $U, V$** but **convex when one is fixed**. Standard solvers:

- **Alternating least squares (ALS)**: fix $V$, solve for $U$ in closed form (it's a per-user least squares); fix $U$, solve for $V$. Iterate. Highly parallelizable per user / item.
- **SGD**: sample $(u, i, r)$ at random, update $u_u$ and $v_i$ along the gradient. Easier to extend with side information.

For implicit feedback (clicks, views. No explicit rating), the loss changes:

$$
\min_{U, V} \sum_{u, i} c_{ui} (p_{ui} - u_u^\top v_i)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)
$$

where $p_{ui} = 1$ if the user interacted with $i$, else $0$, and $c_{ui}$ is a confidence weight [(Hu et al., 2008)](http://yifanhu.net/PUB/cf.pdf). Critical: includes all $(u, i)$ pairs (with low confidence for negatives), not just observed positives.

## Cold start

MF works only for users and items observed during training. For new users and items, MF gives no embedding. Workarounds:

- **Hybrid models**: incorporate side information (item features, user demographics).
- **Two-tower neural models**: encoders take features, can embed arbitrary new users/items at inference.
- **Average-of-similar**: until enough interactions accumulate, use content-based similarity.

This is why two-tower models displaced pure MF for production: they handle cold start naturally.

## Properties

- **Embeddings are not interpretable directly**. They live in an arbitrary basis. PCA-rotate them if you want to look.
- **Rank $k$ is the main hyperparameter**: typical 32–512 for production. Too small → underfits; too large → overfits and slow.
- **Implicit dimensions**: $k$ "latent factors" emerge that often correlate with interpretable concepts (genre, popularity, user activity level).

## Connection to two-tower models

Two-tower retrieval (see [two-tower retrieval](/concepts/two-tower-retrieval/)) is exactly MF generalized to neural encoders:

- MF: $u_u, v_i$ are free embedding parameters.
- Two-tower: $u_u = f_\theta(\text{user features})$, $v_i = g_\phi(\text{item features})$.

Training is similar (sampled-softmax loss replaces squared error for retrieval). The neural version handles cold-start, side info, and personalization beyond ID embeddings.

## When to use plain MF in 2026

- Small-scale prototypes, when neural infrastructure is overkill.
- Strong baseline before complex models.
- Pure rating prediction tasks (Netflix Prize style).
- Embedding initialization for downstream models.

For most production systems: two-tower neural models or transformer-based ranking models have replaced MF as the primary architecture.

## Common pitfalls

- **Treating MF as a similarity-based system.** MF embeddings are learned for prediction, not similarity; cosine of MF embeddings is *not* automatically meaningful as user/item similarity.
- **Ignoring biases.** Without user/item biases, MF spends embedding capacity modeling popular-vs-niche, which is better captured by a scalar.
- **Using MF with explicit-feedback loss on implicit data.** The "missing = zero rating" assumption is wrong; use weighted ALS for implicit feedback.
- **Comparing MF accuracy on RMSE alone.** RMSE doesn't capture top-k ranking quality, which is what matters in recsys.

## Related

- [Two-tower retrieval](/concepts/two-tower-retrieval/). Neural generalization.
- [Embedding spaces](/concepts/embedding-spaces-and-similarity/). How the latent factors are used.
