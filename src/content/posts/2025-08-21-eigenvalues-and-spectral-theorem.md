---
title: "Eigenvalues and the spectral theorem"
description: "Eigenvectors are directions a matrix only stretches. The spectral theorem says symmetric matrices have a full orthogonal eigenbasis with real eigenvalues."
date: "2025-08-21"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An **eigenvector** of $A$ is a non-zero vector $v$ such that $A v = \lambda v$ for some scalar $\lambda$ (the **eigenvalue**). $A$ acts on $v$ purely by scaling. The **spectral theorem** states that every real symmetric matrix is orthogonally diagonalizable: $A = Q \Lambda Q^\top$ with $Q$ orthogonal and $\Lambda$ diagonal real.

Eigendecompositions explain stability of dynamical systems, convergence of optimization, structure of covariance matrices, and properties of attention / graph operators. The spectral theorem is the mathematical reason PCA works on covariance matrices, why Laplacian eigenmaps make sense for graphs, and why second-order optimizers reason about Hessian eigenvalues.

## Eigenvalues, eigenvectors, characteristic polynomial

For a square matrix $A \in \mathbb{R}^{n \times n}$:

- $A v = \lambda v$ is equivalent to $(A - \lambda I) v = 0$, so eigenvalues are roots of $\det(A - \lambda I) = 0$ (the characteristic polynomial).
- An $n \times n$ matrix has $n$ eigenvalues (counted with multiplicity), possibly complex, possibly repeated.
- Trace $= \sum_i \lambda_i$. Determinant $= \prod_i \lambda_i$.

For symmetric matrices, **all eigenvalues are real** and there exists an orthonormal eigenbasis.

## The spectral theorem (symmetric case)

If $A = A^\top \in \mathbb{R}^{n \times n}$:

$$
A = Q \Lambda Q^\top = \sum_{i=1}^n \lambda_i q_i q_i^\top
$$

where $Q = [q_1 | \dots | q_n]$ is orthogonal ($Q^\top Q = I$) and $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$.

Geometric meaning: in the eigenbasis $\{q_i\}$, the action of $A$ is independent scaling along each axis. Symmetric matrices have **no rotational component**. They are pure stretches in some orthogonal frame.

## Connection to SVD

For symmetric positive semi-definite $A$: SVD and eigendecomposition coincide ($U = V = Q$, $\Sigma = \Lambda$). For general matrices they differ. SVD is the more general tool; eigendecomposition is the specialized one for symmetric / square matrices.

## Where eigenvalues show up in ML

| Object | What its eigenvalues tell you |
|--------|------------------------------|
| Covariance matrix | Variances along principal axes (PCA) |
| Hessian of loss | Local curvature; condition number = $\lambda_\max / \lambda_\min$ |
| Graph Laplacian | Connectivity, spectral clustering, GNN smoothness |
| Markov transition matrix | Mixing rate (second-largest eigenvalue) |
| Attention $Q K^\top$ | Effective rank; low-rank structure |
| Recurrent weight matrix | Whether RNN gradients explode/vanish |

## Common pitfalls

- **Treating asymmetric matrices like symmetric ones.** Asymmetric matrices may have complex eigenvalues and may not be diagonalizable at all (Jordan form).
- **Computing eigendecomposition for huge matrices.** Use Lanczos / Arnoldi or randomized SVD for large-scale; full eigendecomposition is $O(n^3)$.
- **Confusing eigenvalues with singular values.** Equal only for symmetric PSD matrices; otherwise singular values are $\sqrt{\lambda_i(A^\top A)}$.

## Related

- [SVD and PCA](/concepts/svd-and-pca/). Generalization to all matrices.
- [Positive definite matrices](/concepts/positive-definite-matrices/). The cone of PSD matrices.
