---
title: "Backpropagation"
description: "Reverse-mode automatic differentiation applied to a computation graph. The algorithm that computes gradients for every parameter in one backward pass."
date: "2025-11-06"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Backpropagation** computes gradients of a scalar loss with respect to every parameter in a neural network in one backward pass through the computation graph, by applying the chain rule from the output back to the inputs and reusing intermediate computations.

## Why it matters

Without backprop, training a deep network would require either:

- Numerical differentiation: $O(P)$ forward passes for $P$ parameters. Infeasible.
- Forward-mode autodiff: $O(P)$ as well; works for small parameter counts but not for neural nets.

Backprop computes all $P$ gradients in $O(F)$ time where $F$ is the cost of a forward pass. Typically 2–3× the forward cost. This is the algorithmic enabler of all modern deep learning.

## The algorithm

Compute the loss as a function of inputs and parameters by composing simple operations: $L = f_n(f_{n-1}(\dots f_1(x; \theta_1) \dots ; \theta_{n-1}); \theta_n)$. Each $f_i$ has known local derivatives.

**Forward pass**: compute and store $z_1, z_2, \dots, z_n = L$ along the way. The intermediates $z_i$ (activations) are needed for backward.

**Backward pass**: starting from $\partial L / \partial L = 1$, recursively apply:

$$
\frac{\partial L}{\partial z_{i-1}} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial z_{i-1}}, \quad
\frac{\partial L}{\partial \theta_i} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial \theta_i}.
$$

The "gradient w.r.t. $z_i$" is the **upstream gradient**; the local Jacobian $\partial z_i / \partial \cdot$ is multiplied in (as a vector-Jacobian product, never materialized as a full matrix).

## Vector-Jacobian products (VJPs)

For an op $z_{i+1} = f(z_i)$ where both $z_i, z_{i+1}$ are vectors, the Jacobian $J$ would be enormous. Backprop computes only the **VJP**: $g_i = J^\top g_{i+1}$ where $g_{i+1} = \partial L / \partial z_{i+1}$.

Each elementary op has a hand-coded VJP rule. Frameworks (PyTorch, JAX, TensorFlow) compose them automatically.

## Memory cost

Backprop must store all forward activations until the backward pass uses them. Memory is proportional to the depth of the network times batch size times activation size. Often dominating GPU memory in deep transformer training.

Mitigations:

- **[Activation checkpointing](/concepts/activation-checkpointing/)**: recompute selected activations during backward instead of storing.
- **Mixed precision**: store activations in BF16 instead of FP32.
- **Sequence packing** + smaller batch.

## Connection to reverse-mode autodiff

Backprop is reverse-mode autodiff applied to scalar-output, vector-input functions ($L: \mathbb{R}^P \to \mathbb{R}$). Reverse mode is efficient when output dimension $\ll$ input dimension; for the opposite ($P$ inputs, $M$ outputs with $M \gg P$), forward mode is preferred. Neural network gradients always have $M = 1$, so reverse is the right choice.

## What backprop does NOT do

- It is **not learning**. Backprop computes gradients; SGD / Adam uses them to update parameters.
- It is **not specific to neural networks**. Any composition of differentiable ops with a scalar output can be backpropagated through.
- It does **not enforce convergence**. The gradient may point downhill, but optimization may still get stuck.

## Common pitfalls

- **Calling `loss.backward()` twice without `retain_graph=True`.** Backward frees the graph by default; second call fails.
- **Forgetting `optimizer.zero_grad()`.** Gradients accumulate by default; not zeroing means each step uses the sum of all past gradients (unintended, breaks convergence).
- **`detach()` errors.** Tensors `.detach()`'d from the graph have no gradient; using them where you wanted gradients to flow gives subtle wrong learning.
- **Memory leaks from holding loss tensors.** Keeping references to loss objects keeps the entire computation graph alive; use `loss.item()` for logging.
- **Confusing `requires_grad` with `is_leaf`.** Parameters are typically leaves with `requires_grad=True`; intermediate tensors are non-leaves with `requires_grad=True` because they depend on params.

## Related

- [Matrix calculus](/concepts/matrix-calculus/). The chain rule formalism.
- [Activation checkpointing](/concepts/activation-checkpointing/). Memory optimization for backprop.
