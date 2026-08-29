---
title: "Matrix calculus for ML"
description: "Gradients, Jacobians, and Hessians for vector- and matrix-valued functions. The minimum needed to derive backprop and second-order methods."
date: "2026-02-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Matrix calculus extends the ordinary derivative to functions whose inputs and/or outputs are vectors or matrices. The two organizing objects are the **gradient** (for scalar-valued functions) and the **Jacobian** (for vector-valued functions); the **Hessian** is the matrix of second partial derivatives.

Every learning algorithm computes derivatives of a scalar loss with respect to vector- or tensor-valued parameters. Knowing the right shapes and conventions saves hours of debugging. Backpropagation is matrix calculus applied recursively through a computation graph.

## The four shape combinations

| $f: \mathbb{R}^n \to \mathbb{R}$ | $f: \mathbb{R}^n \to \mathbb{R}^m$ |
|---|---|
| **Gradient** $\nabla f \in \mathbb{R}^n$ | **Jacobian** $J_f \in \mathbb{R}^{m \times n}$, $(J_f)_{ij} = \partial f_i / \partial x_j$ |
| Second derivative: **Hessian** $\nabla^2 f \in \mathbb{R}^{n \times n}$, $(\nabla^2 f)_{ij} = \partial^2 f / \partial x_i \partial x_j$ | (rarely used; tensor-valued) |

Two competing conventions exist:

- **Numerator layout** (used in physics, calculus textbooks): $\nabla f$ is a column vector matching $x$.
- **Denominator layout** (used in stats, ML): same; modern ML universally uses gradient = column vector with the same shape as the parameter.

For matrix parameters $W \in \mathbb{R}^{m \times n}$, the gradient $\partial L / \partial W \in \mathbb{R}^{m \times n}$ has the same shape as $W$.

## Identities you actually use

| Function | Gradient |
|---------|---------|
| $f(x) = a^\top x$ | $\nabla f = a$ |
| $f(x) = x^\top A x$, $A$ symmetric | $\nabla f = 2 A x$ |
| $f(x) = \|x\|_2^2$ | $\nabla f = 2x$ |
| $f(W) = a^\top W b$ | $\partial f / \partial W = a b^\top$ |
| $f(W) = \|W\|_F^2 = \mathrm{tr}(W^\top W)$ | $\partial f / \partial W = 2 W$ |
| $f(W) = \log \det W$ ($W$ PD) | $\partial f / \partial W = W^{-\top}$ |
| $f(W) = \mathrm{tr}(A W)$ | $\partial f / \partial W = A^\top$ |

## Chain rule

For $L(\theta) = g(f(\theta))$:

- Scalar $\to$ scalar $\to$ scalar: $\frac{dL}{d\theta} = g'(f(\theta)) \cdot f'(\theta)$.
- Vector $\to$ vector $\to$ scalar: $\nabla_\theta L = J_f^\top \cdot \nabla_y g$ where $y = f(\theta)$.

Backprop is just this chain rule applied to a computation graph layer-by-layer, with the **Jacobian-vector product** computed implicitly to avoid materializing the full Jacobian.

## Hessian and second-order methods

The Hessian $\nabla^2 L$ describes local curvature. Second-order methods (Newton's method, K-FAC, Shampoo) use it to scale gradients by inverse curvature: $\theta \leftarrow \theta - (\nabla^2 L)^{-1} \nabla L$.

In modern deep learning, the Hessian is too large to materialize ($O(P^2)$ for $P$ parameters). Approximations:

- **Diagonal**: keep only diagonal entries (RMSProp, Adam's $v_t$ approximates this).
- **Block-diagonal**: per-layer Fisher information (K-FAC).
- **Hessian-vector products**: $\nabla^2 L \cdot v$ via two backward passes. Used in conjugate gradient, influence functions.

## Common pitfalls

- **Mismatched layout conventions.** Always check whether your reference uses numerator or denominator layout; the difference is a transpose.
- **Treating gradients as having the same shape as the loss.** They have the shape of the *parameter*, not the loss.
- **Computing Jacobians explicitly.** For a vector-to-vector function with $m, n$ both large, the Jacobian is $mn$ entries. Use vector-Jacobian products via `torch.autograd.grad` or `jax.vjp` instead.
- **Forgetting bias dimensions.** $W x + b$ has $\partial L / \partial b = \sum_i \partial L / \partial y_i$ summed over the batch dimension.
