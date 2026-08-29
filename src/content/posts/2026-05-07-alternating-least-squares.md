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
