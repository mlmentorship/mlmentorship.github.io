---
title: "Matrix factorization for recsys"
description: "Decompose the user-item interaction matrix into user and item embeddings whose dot product approximates the rating. The original collaborative filtering."
date: "2026-02-16"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Matrix factorization** for collaborative filtering [(Koren et al., 2009)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) factorizes a sparse user-item rating matrix $R \in \mathbb{R}^{m \times n}$ into two low-rank matrices: $R \approx U V^\top$ where $U \in \mathbb{R}^{m \times k}$ contains user embeddings and $V \in \mathbb{R}^{n \times k}$ contains item embeddings. Predicted rating: $\hat r_{ui} = u_u^\top v_i$.

## Why it matters

MF was the dominant collaborative-filtering method from the Netflix Prize era (2006–2009) through about 2018. It still underlies modern two-tower retrieval, embedding-based recsys, and has clean equivalences to many later techniques (matrix completion, neural MF, etc.). Knowing MF makes the move to two-tower neural models obvious.

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
