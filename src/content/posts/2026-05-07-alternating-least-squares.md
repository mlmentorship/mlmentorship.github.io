---
title: "Alternating least squares for collaborative filtering"
description: "Factorize the user-item matrix into two low-rank factors. Each is a linear regression given the other, so alternate. The classical recsys workhorse before deep learning."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Alternating Least Squares** (ALS) factorizes a sparse rating matrix $R \approx U V^\top$ where $U \in \mathbb{R}^{m \times k}$ holds user factors and $V \in \mathbb{R}^{n \times k}$ holds item factors. Optimization alternates: fix $V$, solve for $U$ in closed form (a linear regression per user); fix $U$, solve for $V$. Repeat.

The classic Netflix Prize era was largely won by matrix factorization, and ALS is the simplest training algorithm for it. SGD-based factorization is competitive on dense data, but ALS dominates when the data is implicit-feedback or stored row- and column-blocked across a cluster (Spark MLlib's recommender is ALS).

ALS is still the right baseline for any recommender system before you reach for two-tower retrieval or sequence models. Cheap to train, easy to parallelize, well-understood failure modes.

## The mechanism

Loss with regularization:

$$
\mathcal{L}(U, V) = \sum_{(i, j) \in \Omega} (R_{ij} - u_i^\top v_j)^2 + \lambda \left( \sum_i \|u_i\|^2 + \sum_j \|v_j\|^2 \right),
$$

where $\Omega$ is the set of observed ratings.

Fix all $v_j$. The loss in $u_i$ is a ridge regression:

$$
u_i = \left( \sum_{j \in \Omega_i} v_j v_j^\top + \lambda I \right)^{-1} \sum_{j \in \Omega_i} R_{ij} v_j.
$$

A $k \times k$ system per user. Solve for all $m$ users in parallel. Then fix $U$ and solve for each $v_j$ symmetrically. Iterate until convergence.

<!-- visual:als-alternating-independent-solves -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="als-visual-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="als-visual-title">Freezing one dense factor separates the other factor into independent row solves.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 380" role="img" aria-labelledby="als-svg-title als-svg-desc">
			<title id="als-svg-title">Alternating independent least-squares updates for matrix factorization</title>
			<desc id="als-svg-desc">A sparse m by n ratings matrix is approximated by a dense m by k user-factor matrix times a dense k by n transposed item-factor matrix. In step one, the item factors are fixed while all m user vectors are solved independently. In step two, the user factors are fixed while all n item vectors are solved independently. An arrow returns from step two to step one to show the alternating cycle.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="24" y="20">MODEL: observed ratings ≈ user factors × item factors</text>
			<rect class="viz-node viz-node--input" x="30" y="45" width="160" height="125" rx="10"></rect>
			<text class="viz-node-label" x="110" y="69">Sparse ratings R</text>
			<text class="viz-node-value" x="110" y="87">m users × n items</text>
			<path class="viz-gridline" d="M54 99 H166 M54 119 H166 M54 139 H166 M76 92 V151 M99 92 V151 M122 92 V151 M145 92 V151"></path>
			<text class="viz-node-value" x="65" y="110">5</text>
			<text class="viz-node-value" x="111" y="130">2</text>
			<text class="viz-node-value" x="88" y="150">4</text>
			<text class="viz-node-value" x="157" y="110">3</text>
			<text class="viz-node-label" x="228" y="113">≈</text>
			<rect class="viz-node viz-node--focus" x="265" y="45" width="105" height="125" rx="10"></rect>
			<text class="viz-node-label" x="317" y="91">U</text>
			<text class="viz-node-value" x="317" y="111">m × k</text>
			<text class="viz-node-value" x="317" y="132">one row uᵢ per user</text>
			<text class="viz-node-label" x="403" y="113">×</text>
			<rect class="viz-node viz-node--output" x="440" y="70" width="270" height="75" rx="10"></rect>
			<text class="viz-node-label" x="575" y="96">Vᵀ</text>
			<text class="viz-node-value" x="575" y="116">k × n · one column vⱼ per item</text>
			<path class="viz-forward" d="M317 171 V221"></path>
			<path class="viz-forward" d="M575 146 C575 184 592 196 592 221"></path>
			<text class="viz-axis-label" x="24" y="218">OPTIMIZATION: alternate which factor is fixed</text>
			<rect class="viz-node viz-node--focus" x="30" y="235" width="310" height="95" rx="12"></rect>
			<text class="viz-node-label" x="185" y="260">1 · FIX V, SOLVE U</text>
			<text class="viz-node-value" x="185" y="282">u₁ system  ‖  u₂ system  ‖  …  ‖  uₘ system</text>
			<text class="viz-node-value" x="185" y="304">m independent ridge-regression solves</text>
			<rect class="viz-node viz-node--output" x="420" y="235" width="310" height="95" rx="12"></rect>
			<text class="viz-node-label" x="575" y="260">2 · FIX U, SOLVE V</text>
			<text class="viz-node-value" x="575" y="282">v₁ system  ‖  v₂ system  ‖  …  ‖  vₙ system</text>
			<text class="viz-node-value" x="575" y="304">n independent ridge-regression solves</text>
			<path class="viz-forward" d="M341 282 H410"></path>
			<path class="viz-forward" d="M575 331 C575 370 185 370 185 340"></path>
			<text class="viz-edge-label" x="380" y="355">repeat until the loss stops improving</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> hold every item vector fixed and no user solve depends on another user; then hold every user vector fixed and no item solve depends on another item. Those parallel row solves are why a jointly non-convex factorization becomes tractable one side at a time.</figcaption>
</figure>

The objective is **biconvex**: convex in $U$ given $V$ and convex in $V$ given $U$, but not jointly convex. ALS finds a local minimum, which is empirically good on real recsys data.

## Implicit feedback (the practical version)

In real systems, ratings are rare. What you have is implicit signal: clicks, watches, plays. Treat all observed interactions as positives and all missing entries as weak negatives. Hu et al. ([2008](http://yifanhu.net/PUB/cf.pdf)) reformulated ALS for this:

Replace $R_{ij}$ with a binary preference $p_{ij} \in \{0, 1\}$ and a confidence weight $c_{ij} = 1 + \alpha r_{ij}$ where $r_{ij}$ is the observed interaction count.

$$
\mathcal{L} = \sum_{i, j} c_{ij} (p_{ij} - u_i^\top v_j)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2).
$$

The sum is now over **all** entries, not just observed. The closed-form ALS step still works because the per-user system can be rewritten as

$$
u_i = (V^\top C^i V + \lambda I)^{-1} V^\top C^i p_i,
$$

with the trick that $V^\top C^i V = V^\top V + V^\top (C^i - I) V$. The first term is precomputed and shared across users; the second is sparse.

## Bias terms

Real ratings have systematic shifts: some users rate high, some low; some items are universally loved. Add bias terms:

$$
\hat{R}_{ij} = \mu + b_i + b_j + u_i^\top v_j,
$$

where $\mu$ is the global mean, $b_i$ the user bias, $b_j$ the item bias. Biases are also learned in the same alternating framework.

## Tradeoffs vs alternatives

| Method | Pros | Cons |
|---|---|---|
| **ALS** | Closed-form per step, parallelizable, no learning rate | $O(k^3)$ per user; large $k$ is expensive |
| **SGD on factorization** | Tiny memory, online-friendly | Needs LR tuning, slower wall-clock at scale |
| **Two-tower neural** | Cold-start via features, content awareness | Needs more data, harder to train |
| **BPR / pairwise loss** | Better implicit-feedback ranking | Not closed-form, needs negative sampling |

For a fresh recsys project at moderate scale: ALS first, two-tower if you need cold-start handling or richer features.

## Common pitfalls

- **Treating all missing entries as negatives without confidence weighting**. A user not interacting with an item could be a negative or just unseen. Confidence weighting in implicit ALS handles this.
- **Choosing $k$ too large**. Latent factors of 50 to 200 are typical; bigger $k$ overfits and is slower.
- **Forgetting to regularize**. Without $\lambda$, ALS overfits trivially on observed entries.
- **Comparing to baselines that include bias terms while yours does not**. Always include $\mu + b_i + b_j$ before declaring an improvement.
- **Running ALS on truly massive data without distributed setup**. Spark and similar systems exist exactly for this.

## Related

- [Matrix factorization for recsys](/concepts/matrix-factorization-recsys/).
- [Two-tower retrieval](/concepts/two-tower-retrieval/).
- [SVD and PCA](/concepts/svd-and-pca/).
