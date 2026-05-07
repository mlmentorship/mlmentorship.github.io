---
title: "Matrices as linear maps"
description: "A matrix is a linear function from one vector space to another. Every operation in ML. Projection, rotation, basis change, gradient flow. Is matrix multiplication."
date: "2025-09-15"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A matrix $A \in \mathbb{R}^{m \times n}$ represents a linear map $f: \mathbb{R}^n \to \mathbb{R}^m$ defined by $f(x) = Ax$. Composition of linear maps corresponds to matrix multiplication; the columns of $A$ are the images of the standard basis vectors.

## Why it matters

Every layer in a neural network is a linear map (followed by a non-linearity). Every embedding lookup, every attention score, every gradient backward pass is a matrix multiplication. Understanding what a matrix *does* geometrically. Rather than just how to compute with it. Is the foundation for reasoning about model capacity, conditioning, and gradient flow.

## The geometry

For $A \in \mathbb{R}^{m \times n}$:

- **Columns of $A$** = images of $e_1, \dots, e_n$. Span them and you get the **column space** (range of the map).
- **Rows of $A$** = linear functionals; span the **row space**.
- **Null space** = $\{x : Ax = 0\}$. Directions the map collapses.
- **Rank** = dimension of column space = dimension of row space.

If $A$ is square and invertible, $A$ is a bijection: it stretches, rotates, and reflects $\mathbb{R}^n$ without losing information. If rank $< n$, $A$ collapses dimensions.

## Composition and multiplication

If $f(x) = Ax$ and $g(y) = By$, then $g \circ f(x) = B(Ax) = (BA)x$. Matrix multiplication is the composition of linear maps. This is *why* multiplication is associative ($f \circ (g \circ h) = (f \circ g) \circ h$) but not commutative (order of operations matters).

## Special families

| Matrix | Geometric action |
|--------|----------------|
| Orthogonal $Q$ ($Q^\top Q = I$) | Rotation or reflection (preserves length and angle) |
| Diagonal | Independent scaling along each axis |
| Symmetric | Has real eigenvalues; orthogonal eigenvector basis |
| Positive definite | Symmetric + all eigenvalues > 0; defines an inner product |
| Permutation | Reorders coordinates |
| Projection ($P^2 = P$) | Maps onto a subspace, kills orthogonal complement |

## Common pitfalls

- **Treating matrix multiplication as element-wise.** Use Hadamard ($\odot$) for element-wise; matrix multiplication is composition.
- **Forgetting that shapes determine the map.** $A \in \mathbb{R}^{3 \times 5}$ is a map $\mathbb{R}^5 \to \mathbb{R}^3$, not the other way around.
- **Confusing column space with row space.** Both have dimension = rank, but they live in different spaces ($\mathbb{R}^m$ vs $\mathbb{R}^n$).
