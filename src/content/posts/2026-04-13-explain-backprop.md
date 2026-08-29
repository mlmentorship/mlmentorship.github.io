---
title: "Explain backprop in your own words"
description: "Backprop is reverse-mode automatic differentiation. It reuses forward-pass values to compute all parameter gradients at roughly one additional forward-pass cost."
date: "2026-04-13"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at every level.*

The interviewer is checking whether you understand backprop as a *computation*, not as a formula. The L4 answer recites the chain rule. The L6 answer connects it to reverse-mode auto-differentiation, the cost relative to forward, and what breaks it in practice.

## What an L4 answer sounds like

> "Backprop computes gradients of the loss with respect to every weight using the chain rule. Starting from the output, you propagate the error backwards through each layer."

Correct, but generic. You've memorized the textbook line, not internalized the algorithm.

## What an L5 answer sounds like

> "Backprop is reverse-mode automatic differentiation applied to the computation graph of a neural network.
>
> The forward pass computes the loss while caching intermediate activations. The backward pass starts at the loss with gradient 1, then walks backward through the cached graph. At each operation, it multiplies the upstream gradient by the local Jacobian of that op with respect to its inputs (the chain rule), and accumulates gradients at parameters.
>
> The key efficiency insight: reverse-mode evaluates one forward and one backward pass per output. A naive 'compute gradient by changing each parameter' approach would cost O(P) forward passes for P parameters. Reverse-mode is O(1) forward + O(1) backward to get *all* P gradients."

This is L5. You've explained it as auto-differentiation, mentioned the activation caching, and quantified the efficiency win.

## What an L6 answer sounds like

> "...a few practical things worth adding:
>
> **Memory is the cost.** The backward pass needs the cached activations from the forward pass. For deep networks with large activations, this dominates GPU memory. Activation checkpointing trades compute for memory by recomputing activations during backward instead of storing them.
>
> **Backprop is exact, not approximate.** Unlike numerical differentiation, it has no truncation error. The only error is floating-point.
>
> **It composes through any differentiable op.** This is why frameworks (PyTorch, JAX) implement autograd as a graph of primitive ops with known local Jacobians. Custom ops just need to define forward and the local Jacobian (vector-Jacobian product); the framework composes the rest.
>
> **What breaks it in practice**: non-differentiable ops (argmax, hard threshold), vanishing gradients in deep networks (ReLU, residuals, normalization mitigate), exploding gradients (gradient clipping mitigates), and detached tensors silently breaking gradient flow."

## Tells that get you a strong-hire vote

- You frame it as **reverse-mode auto-diff**, not just "the chain rule."
- You mention **activation caching** and the memory cost.
- You connect to **activation checkpointing** as the standard memory-compute trade.
- You distinguish backprop (exact) from numerical differentiation.
- You mention **detached tensors** as a common bug source.

## Tells that get you down-leveled

- "It's the chain rule" with no further detail.
- Confusion about which gradients flow where (e.g., gradients flow through cached activations, not just weights).
- No awareness of the memory cost.
- Calling it "back-propagation of errors" without explaining what's actually being computed.

## Common follow-up

"Why is reverse-mode preferred over forward-mode auto-diff for neural networks?"

The L6 answer:

> "Forward-mode computes the derivative of *one input* with respect to *all outputs* in one pass. Reverse-mode computes the derivative of *one output* (the loss) with respect to *all inputs* (parameters) in one pass. Neural network training has many parameters and one scalar loss, so reverse-mode is the right shape. For small models with many outputs and few inputs, forward-mode can be faster."

---

*Related: [Adam and AdamW](/concepts/adam-and-adamw/) and [debug a model that is not learning](/questions/debug-model-not-learning/).*
