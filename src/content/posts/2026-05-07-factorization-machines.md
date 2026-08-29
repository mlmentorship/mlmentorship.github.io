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
