---
title: "Determinant and volume"
description: "The determinant of a matrix is the signed volume scaling factor of the linear map. Zero determinant means the map collapses dimensions."
date: "2025-12-02"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

The determinant $\det(A)$ of a square matrix $A$ is the signed factor by which $A$ scales $n$-dimensional volumes. Geometrically, $|\det(A)|$ is the volume of the parallelepiped spanned by $A$'s column vectors; the sign flips if $A$ contains a reflection.

Determinants tell you whether a linear map is invertible (non-zero det) or singular (zero det). They appear in change-of-variables formulas (probability density transformations, normalizing flows), in computing volumes of parallelotopes (Gaussian likelihoods), and as Jacobian determinants in differential geometry.

## Properties

For $A, B \in \mathbb{R}^{n \times n}$:

- $\det(AB) = \det(A) \det(B)$. Composition multiplies volumes.
- $\det(A^\top) = \det(A)$.
- $\det(A^{-1}) = 1/\det(A)$ when $A$ is invertible.
- $\det(cA) = c^n \det(A)$ for scalar $c$.
- $\det = 0 \iff$ columns linearly dependent $\iff$ $A$ not invertible $\iff$ $A$ has a zero eigenvalue.
- $\det(A) = \prod_i \lambda_i$ (product of eigenvalues, with multiplicity, possibly complex).
- For triangular matrices: $\det = \prod_i A_{ii}$ (product of diagonal).

## Geometric interpretation

The columns of $A$ are vectors in $\mathbb{R}^n$. They span a parallelepiped. Its volume is $|\det(A)|$.

- $\det(A) > 0$: $A$ preserves orientation.
- $\det(A) < 0$: $A$ reverses orientation (contains a reflection).
- $\det(A) = 0$: parallelepiped is flat. Columns are linearly dependent. $A$ collapses at least one dimension.

For an orthogonal matrix $Q$: $|\det(Q)| = 1$ (rotation/reflection preserves volume).

## Change of variables (probability)

If $Y = f(X)$ with $f$ invertible and differentiable, the density of $Y$ is

$$
p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)|
$$

where $J_{f^{-1}}$ is the Jacobian of the inverse transform. This is the basis of normalizing flows: pick $f$ so the Jacobian determinant is cheap to compute (triangular Jacobian → $\det = \prod_i \partial y_i / \partial x_i$).

## Computing determinants

| Method | Cost | When |
|--------|------|------|
| LU decomposition | $O(n^3)$ | General-purpose; standard library default |
| Triangular: product of diag | $O(n)$ | When matrix is already triangular |
| Eigendecomposition: $\prod \lambda_i$ | $O(n^3)$ | If you need eigenvalues anyway |
| **Log-determinant** for PD matrices | via Cholesky | When you only need $\log \det$ (e.g., Gaussian log-likelihood) |

Numerical tip: for large matrices, compute $\log \det$ directly (sum of $\log$ of LU diagonal); $\det$ itself overflows or underflows quickly.

## Common pitfalls

- **Using $\det$ as a proxy for matrix "size."** A nearly singular matrix can have huge entries but tiny $\det$.
- **Computing $\det$ for invertibility tests.** Numerically unstable; use rank or condition number instead.
- **Forgetting the absolute value in change-of-variables formulas.** Densities are non-negative; the Jacobian determinant can be negative.
