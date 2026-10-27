---
title: "Matrix factorization for recsys"
description: "Predict missing user-item ratings from shared factors, while separating observed targets, implicit-feedback assumptions, and cold-item support."
date: "2026-02-16"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A missing rating supplies two indices: a user row and an item row.
Matrix factorization shares learned rows across many observed ratings rather than storing an independent prediction per cell.
Its low-rank assumption is a modeling choice, not a guarantee of matrix completion.

Let $R\in\mathbb R^{m\times n}$, $U\in\mathbb R^{m\times k}$, and $V\in\mathbb R^{n\times k}$.
The shared coordinate count is $k$, and $R\approx UV^\top$.
With zero biases, fixture rows $u_A=(2,1)$ and $v_3=(1.5,1)$ predict $2(1.5)+1(1)=4$.
These are chosen vectors, not a fitted solution for all displayed ratings.

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
	<figcaption><strong>Read it this way:</strong> user A selects row A of <var>U</var>; item 3 selects row 3 of <var>V</var>. Matching-coordinate products sum to four. The question mark is unknown, not a zero training target.</figcaption>
</figure>

| Entry in user A's row | Observation | Explicit-rating loss |
| --- | --- | --- |
| Item 1 | Rating 5 | Included |
| Item 3 | Unknown | Excluded |
| Item 4 | Rating 1 | Included |

## The objective

Minimize regularized squared error on observed ratings $\Omega = \{(u, i) : r_{ui} \text{ observed}\}$:

$$
\min_{U, V} \sum_{(u, i) \in \Omega} (r_{ui} - u_u^\top v_i)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2).
$$

The observed set $\Omega$ excludes unknown entries, and $\lambda\ge0$ controls the stated L2 penalty.
With biases, use $\hat r_{ui}=\mu+b_u+b_i+u_u^\top v_i$, defining global mean and user/item offsets.
Fit these together; adding $\mu=3,b_A=0.2,b_3=-0.1$ afterward changes four to $7.1$.
That exceeds a 1-5 rating scale and is not automatically a calibrated prediction.

### Change observed signal

**New task:** replace explicit ratings with clicks whose exposure is not fully observed.

**Picture and change:** a confidence-weighted implicit model uses

$$
\min_{U, V} \sum_{u, i} c_{ui} (p_{ui} - u_u^\top v_i)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)
$$

Here $p_{ui}$ is an interaction indicator and $c_{ui}\ge0$ is modeling confidence, not an identified exposure probability.
Unlike explicit-rating loss, this objective includes all pairs, usually with lower confidence for unobserved interactions.

**Check:** a separate two-item fixture has scores $(0.8,0.2)$, preferences $(1,0)$, and confidences $(5,1)$.
Its unregularized loss is $5(1-0.8)^2+(0-0.2)^2=0.24$, not the positive-only value $0.20$.
A never-shown item's weak zero preference remains an assumption, not evidence of dislike.

**Use when:** revise the objective and evaluation when observation semantics change.

## Cold start

### Change item availability

**New task:** score a new item without a trained ID row.

**Picture and change:** use a separately validated feature encoder or content/context fallback, rather than a nonexistent lookup row.

**Check:** if an encoder supplies $(1,0)$, user A's zero-bias score is $2(1)+1(0)=2$.
Increasing $k$ alone cannot produce a trained row for an unseen ID.
An encoder can compute a vector without proving cold-item accuracy.

**Use when:** evaluate the cold-item slice and fallback separately from seen-ID performance.

<!-- learning:core-end -->

<div data-learning-extension>

## Training

Squared-error factorization is nonconvex jointly, but convex in either factor when the other is fixed.
[ALS](/concepts/alternating-least-squares/) solves conditional ridge problems; SGD updates selected rows.
For one pair's squared residual and two-row L2 penalty, hold biases fixed and save both old rows.
Set $e=r-\mu-b_u-b_i-u_{\rm old}^\top v_{\rm old}$, then compute simultaneously

$$
u_{\rm new}=(1-2\eta\lambda)u_{\rm old}+2\eta e v_{\rm old},
\qquad
v_{\rm new}=(1-2\eta\lambda)v_{\rm old}+2\eta e u_{\rm old}.
$$

Using the updated user row for the item gradient changes this step.
Sampling and regularization scaling must match the full objective.

## Properties

For orthogonal $Q$, replacing $(U,V)$ with $(UQ,VQ)$ preserves predictions and L2 norms.
The fixture rows can become $(1,-2)$ and $(1,-1.5)$ while retaining dot product four.
Tune $k$ and regularization against matched baselines; coordinates are not uniquely identifiable tastes.

## Connection to two-tower models

[Two-tower retrieval](/concepts/two-tower-retrieval/) replaces free rows with feature encoders $f_\theta,g_\phi$.
An ID-only tower still lacks trained parameters for new IDs.
Sampled retrieval losses differ from squared-error ratings and all-pair implicit fitting.

## When to use plain MF in 2026

MF can be a rating baseline, prototype, or embedding initializer.
Choose architecture from task, latency, features, and measured validation, not a universal deployment-prevalence claim.

## Common pitfalls

Dot-product prediction does not validate within-type cosine; normalization changes scores.
Compare bias baselines and ranking metrics when top-K quality matters; RMSE answers a different question.

Can a PCA rotation identify unique causal tastes?

<details><summary>Answer</summary>

No. It chooses coordinates without removing equivalent factorizations or establishing causal meanings.

</details>

References: [Koren et al. (2009)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf), [Hu et al. (2008)](http://yifanhu.net/PUB/cf.pdf).

</div>

## Related

- [Two-tower retrieval](/concepts/two-tower-retrieval/). Neural generalization.
- [Embedding spaces](/concepts/embedding-spaces-and-similarity/). How the latent factors are used.
