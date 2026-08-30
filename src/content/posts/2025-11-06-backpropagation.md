---
title: "Backpropagation"
description: "Reverse-mode automatic differentiation applied to a computation graph. The algorithm that computes gradients for every parameter in one backward pass."
date: "2025-11-06"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Backpropagation** computes gradients of a scalar loss with respect to every parameter in a neural network in one backward pass through the computation graph, by applying the chain rule from the output back to the inputs and reusing intermediate computations.

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

<!-- visual:backprop-forward-reverse-trace -->
<figure class="learning-figure backprop-visual" aria-labelledby="backprop-visual-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="backprop-visual-title">Values move forward once. Gradients move backward once and reuse the saved values.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 700 300" overflow="hidden" role="img" aria-labelledby="backprop-svg-title backprop-svg-desc">
			<title id="backprop-svg-title">Forward and backward passes through a scalar computation graph</title>
			<desc id="backprop-svg-desc">Inputs x equals 2 and w equals 3 flow through multiplication, addition, and squaring to produce loss 49. A reverse pass starts with loss gradient 1 and applies local derivatives to produce gradients 42 for x and 28 for w.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="24" y="20">FORWARD: compute and cache values</text>
			<path class="viz-forward" d="M102 74 C145 74 157 105 193 122"></path>
			<path class="viz-forward" d="M102 184 C145 184 157 155 193 138"></path>
			<path class="viz-forward" d="M247 130 H353"></path>
			<path class="viz-forward" d="M407 130 H513"></path>
			<text class="viz-edge-label" x="145" y="72">multiply</text>
			<text class="viz-edge-label" x="300" y="118">+ 1</text>
			<text class="viz-edge-label" x="460" y="118">square</text>
			<circle class="viz-node viz-node--input" cx="75" cy="74" r="27"></circle>
			<text class="viz-node-label" x="75" y="69">x</text><text class="viz-node-value" x="75" y="86">value 2</text>
			<circle class="viz-node viz-node--input" cx="75" cy="184" r="27"></circle>
			<text class="viz-node-label" x="75" y="179">w</text><text class="viz-node-value" x="75" y="196">value 3</text>
			<circle class="viz-node viz-node--focus" cx="220" cy="130" r="27"></circle>
			<text class="viz-node-label" x="220" y="125">z₁</text><text class="viz-node-value" x="220" y="142">value 6</text>
			<circle class="viz-node viz-node--focus" cx="380" cy="130" r="27"></circle>
			<text class="viz-node-label" x="380" y="125">z₂</text><text class="viz-node-value" x="380" y="142">value 7</text>
			<circle class="viz-node viz-node--output" cx="540" cy="130" r="27"></circle>
			<text class="viz-node-label" x="540" y="125">L</text><text class="viz-node-value" x="540" y="142">value 49</text>
			<text class="viz-axis-label" x="24" y="250">BACKWARD: upstream gradient × local derivative</text>
			<path class="viz-backward" d="M520 153 C485 204 445 204 401 153"></path>
			<path class="viz-backward" d="M359 153 C325 204 286 204 241 153"></path>
			<path class="viz-backward" d="M199 153 C165 219 126 223 94 203"></path>
			<path class="viz-backward" d="M193 142 C150 153 124 124 94 96"></path>
			<text class="viz-gradient-label" x="460" y="211">1 × 2z₂ = 14</text>
			<text class="viz-gradient-label" x="300" y="211">14 × 1 = 14</text>
			<text class="viz-gradient-label" x="139" y="235">14 × x = 28</text>
			<text class="viz-gradient-label" x="137" y="132">14 × w = 42</text>
			<text class="viz-node-gradient" x="75" y="36">∂L/∂x = 42</text>
			<text class="viz-node-gradient" x="75" y="225">∂L/∂w = 28</text>
			<text class="viz-node-gradient" x="220" y="93">∂L/∂z₁ = 14</text>
			<text class="viz-node-gradient" x="380" y="93">∂L/∂z₂ = 14</text>
			<text class="viz-node-gradient" x="540" y="93">seed = 1</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> each reverse arrow multiplies one upstream gradient by one local derivative. The graph is traversed once in each direction; no full Jacobian is built.</figcaption>
</figure>

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
