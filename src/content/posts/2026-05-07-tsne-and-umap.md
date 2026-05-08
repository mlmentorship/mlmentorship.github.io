---
title: "t-SNE and UMAP: nonlinear dimensionality reduction"
description: "Both project high-dimensional data to 2D for visualization by preserving local neighborhoods. Both are easy to misread. Know what they show and what they hide."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**t-SNE** ([van der Maaten & Hinton, 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html)) and **UMAP** ([McInnes et al., 2018](https://arxiv.org/abs/1802.03426)) embed high-dimensional points into 2 or 3 dimensions while preserving local neighborhood structure. The default tools for "what does this embedding space look like" plots.

## Why it matters

Linear projections (PCA) preserve global variance but smear local structure. For high-dimensional embeddings (transformer activations, sentence embeddings, single-cell genomics), the interesting structure is local: which points cluster together, which categories are separable. t-SNE and UMAP optimize for that locally and produce maps that show the cluster structure clearly.

Almost every embedding visualization you have seen in a paper since 2015 is one of these two.

## What t-SNE optimizes

For each high-dimensional point $x_i$, define a probability distribution over neighbors using a Gaussian:

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2 \sigma_i^2)}{\sum_{k \ne i} \exp(-\|x_i - x_k\|^2 / 2 \sigma_i^2)},
$$

with $\sigma_i$ tuned per point so that the entropy of $p_{\cdot \mid i}$ matches a target **perplexity** (typically 30, an effective neighborhood size).

In 2D, define a heavy-tailed (Student-$t$) distribution:

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|y_k - y_l\|^2)^{-1}}.
$$

Minimize the KL divergence $\text{KL}(P \| Q)$ via gradient descent on $y_i$. The heavy tail in $q$ pushes far-apart points further apart, opening visible gaps between clusters.

## What UMAP optimizes

UMAP builds a fuzzy simplicial set (a weighted graph) of the high-dimensional data using each point's $k$ nearest neighbors. It does the same in low dimension and minimizes a cross-entropy between the two graphs. Faster than t-SNE, scales to millions of points, often gives slightly better global structure.

The math is more involved (it involves Riemannian metrics and category theory in the original paper), but operationally UMAP is "t-SNE on a sparse k-NN graph with a different loss."

## What both preserve and what they don't

**Preserve well**:
- Local neighborhood: which points are close to which.
- Cluster identity: separable groups remain separable.

**Do not preserve**:
- **Distances between clusters.** Cluster $A$ being twice as far from cluster $B$ as from cluster $C$ in the t-SNE plot tells you almost nothing about the high-dimensional reality.
- **Cluster sizes.** A small dense cluster and a large diffuse one can render the same size.
- **Densities.** UMAP and t-SNE both equalize density to some extent.

## The hyperparameters that change everything

**t-SNE**:
- **Perplexity** (5 to 50 typical). Effective neighborhood size. Small perplexity captures fine structure; large perplexity captures broader patterns. Always plot multiple perplexities ([Wattenberg et al., 2016](https://distill.pub/2016/misread-tsne/)).
- **Iterations** (1000+). Under-converged plots can show fake structure.
- **Initialization** (random vs PCA). PCA init gives more reproducible global layout.

**UMAP**:
- **n_neighbors** (15 to 50 typical). Local vs global tradeoff.
- **min_dist** (0.0 to 0.5). How tightly points are packed.
- **metric**. Cosine for embeddings, Euclidean for raw features.

## When to use which

| Use case | Tool |
|---|---|
| Datasets up to ~10k points, careful interpretation | t-SNE |
| Datasets above 100k points, speed matters | UMAP |
| Need approximate global structure | UMAP |
| Reproducible plots across runs | UMAP with fixed seed (t-SNE is also seed-dependent but more sensitive) |

## Common pitfalls

- **Reading distance between clusters as meaningful.** It is not.
- **Running with default hyperparameters and never sweeping.** Conclusions can flip with perplexity or n_neighbors.
- **Using t-SNE for downstream features.** It is for visualization only; the embedding is not a meaningful low-dim representation.
- **Forgetting that the seed matters.** Always report it. Cross-check with multiple seeds before drawing conclusions.
- **Using Euclidean distance on raw embeddings**. Most modern embeddings are designed for cosine; pass `metric="cosine"`.

## Related

- [SVD and PCA](/concepts/svd-and-pca/).
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity-metrics/).
