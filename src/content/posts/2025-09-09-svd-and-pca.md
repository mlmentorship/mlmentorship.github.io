---
title: "SVD and PCA"
description: "The singular value decomposition factorizes any matrix into rotation × stretching × rotation. PCA is SVD applied to mean-centered data."
date: "2025-09-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Every real matrix $A \in \mathbb{R}^{m \times n}$ admits the factorization $A = U \Sigma V^\top$ where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal and $\Sigma$ is diagonal with non-negative entries (singular values). **PCA** is SVD applied to a mean-centered data matrix.

## Why it matters

SVD is the universal matrix factorization. It exists for every matrix, even rectangular and rank-deficient ones. Reading off properties from the SVD answers "what does this matrix do?": singular values give scaling factors, $V$ gives input directions, $U$ gives output directions.

PCA is the canonical use of SVD: project data onto the directions of largest variance to get a low-dimensional representation that preserves as much information as possible.

## The decomposition

$A = U \Sigma V^\top$:

- $V$'s columns are an orthonormal basis of the row space of $A$ (input directions).
- $U$'s columns are an orthonormal basis of the column space (output directions).
- $\Sigma$'s diagonal entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ are the singular values (how much each input direction is stretched into its output direction).

Geometrically: any linear map is "rotate the input, stretch axis-by-axis, rotate the output." That's it.

The rank of $A$ is the number of non-zero singular values. The condition number is $\sigma_1 / \sigma_r$ where $r$ is the rank.

## Truncated SVD and low-rank approximation

The best rank-$k$ approximation of $A$ in Frobenius (or spectral) norm is

$$
A_k = U_k \Sigma_k V_k^\top
$$

where $U_k, V_k$ keep the first $k$ columns and $\Sigma_k$ keeps the first $k$ singular values (Eckart–Young theorem). Used in: dimensionality reduction, image compression, embedding regularization, low-rank LoRA fine-tuning.

## PCA as SVD

Given a data matrix $X \in \mathbb{R}^{n \times d}$ ($n$ samples, $d$ features):

1. Mean-center: $\tilde X = X - \bar x$.
2. Compute SVD: $\tilde X = U \Sigma V^\top$.
3. The columns of $V$ are the **principal components** (directions of maximum variance in feature space).
4. The variance along the $i$-th component is $\sigma_i^2 / (n - 1)$.
5. Project to $k$ dimensions: $Z = \tilde X V_k = U_k \Sigma_k$.

Equivalent formulation: PCA = eigendecomposition of the sample covariance $\tilde X^\top \tilde X / (n-1)$. SVD is numerically more stable.

## Common pitfalls

- **Forgetting to center.** PCA on uncentered data finds the direction toward the mean as PC1, which is rarely what you want.
- **Forgetting to scale.** If features have different units, large-magnitude features dominate; standardize (divide by std) before PCA when units differ.
- **Confusing PCA with whitening.** PCA gives uncorrelated components but not unit variance. Whitening = PCA + scale to unit variance.
- **Using PCA on categorical / sparse data without thought.** PCA assumes Euclidean structure; for sparse / categorical data, look at NMF, LDA, or contrastive embeddings.

## Related

- [Matrices as linear maps](/concepts/matrices-as-linear-maps/). The geometry.
- [Eigenvalues and the spectral theorem](/concepts/eigenvalues-and-spectral-theorem/). For symmetric matrices.
