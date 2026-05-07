---
title: "Positive (semi-)definite matrices"
description: "Matrices that define inner products and proper covariances. The geometry of PSD: ellipsoids, not arbitrary shapes."
date: "2026-01-23"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is **positive semi-definite (PSD)** if $x^\top A x \ge 0$ for all $x \in \mathbb{R}^n$, and **positive definite (PD)** if $x^\top A x > 0$ for all $x \ne 0$. Equivalently: all eigenvalues are non-negative (PSD) or strictly positive (PD).

## Why it matters

PSD matrices are the matrices that can serve as **covariance matrices**, **kernel matrices** (Gram matrices), **inner-product weight matrices**, and **Hessians at local minima**. The PSD cone is the natural domain for many optimization problems (semidefinite programming, Gaussian processes, kernel methods).

## Equivalent characterizations

For symmetric $A$:

- $x^\top A x \ge 0$ for all $x$ (definition).
- All eigenvalues $\lambda_i \ge 0$.
- $A = B^\top B$ for some matrix $B$ (factorization, e.g., Cholesky $B = L^\top$ with $L$ lower triangular).
- $A$ is the covariance matrix of some random vector.
- All principal minors (determinants of upper-left $k \times k$ blocks) are non-negative.

For PD: same with strict inequalities everywhere.

## The Cholesky factorization

Every PD matrix has a unique decomposition $A = L L^\top$ with $L$ lower triangular and positive diagonal. This is the standard way to:

- Solve $Ax = b$ when $A$ is PD ($O(n^3/3)$ instead of $O(n^3)$ for general LU).
- Sample from a Gaussian: if $z \sim \mathcal{N}(0, I)$ then $L z \sim \mathcal{N}(0, A)$.
- Compute Gaussian log-likelihoods: $\log \det A = 2 \sum_i \log L_{ii}$.

PSD (not strictly PD) matrices admit Cholesky-like decompositions but with possible zero diagonal entries; use pivoted Cholesky or LDL.

## The PSD cone

The set of $n \times n$ PSD matrices forms a **convex cone** (closed under non-negative combinations). This is why semidefinite programming generalizes linear programming. It optimizes over a different cone.

Operations preserving PSD:

- $A + B$ is PSD if $A, B$ are PSD.
- $c A$ is PSD for $c \ge 0$.
- $B^\top A B$ is PSD for any compatible $B$.
- Element-wise (Hadamard) product (Schur product theorem).

Operations *not* preserving PSD:

- General matrix product $AB$ (only if $A, B$ commute).
- Inverse: PD matrices have PD inverses; PSD with zero eigenvalue is not invertible.

## Geometric intuition

For $A$ PD, the set $\{x : x^\top A x \le 1\}$ is a closed ellipsoid centered at the origin. Eigenvectors of $A$ give the axes; eigenvalues give $1/\text{axis-length}^2$. PSD matrices that are not PD give degenerate ellipsoids (flat in some direction).

## Common pitfalls

- **Calling a non-symmetric matrix PSD.** PSD is defined for symmetric matrices. For asymmetric $A$, the relevant object is $\frac{1}{2}(A + A^\top)$.
- **Trusting numerical eigenvalues at machine precision.** A theoretically PSD covariance computed from data can have tiny negative eigenvalues from rounding. Use jitter ($A + \varepsilon I$) before Cholesky.
- **Confusing PSD with diagonally dominant.** Diagonally dominant with positive diagonal $\Rightarrow$ PSD, but the converse is false.
- **Inverting near-singular PSD matrices.** Always check the smallest eigenvalue or condition number first; regularize if needed.
