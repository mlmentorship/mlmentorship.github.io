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

<!-- visual:explain-backprop-one-loss-all-gradients -->
<figure class="learning-figure backprop-visual" aria-labelledby="explain-backprop-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="explain-backprop-title">How does one scalar loss produce a gradient for every parameter?</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 700 320" role="img" aria-labelledby="explain-backprop-svg-title explain-backprop-svg-desc">
			<title id="explain-backprop-svg-title">One forward pass and one reverse sweep through a three-operation graph</title>
			<desc id="explain-backprop-svg-desc">A solid path carries input x through operations f1, f2, and f3 to scalar loss L. Parameters theta 1, theta 2, and theta 3 each enter one operation. Activations a1 and a2 are explicitly saved during the forward pass. A dashed reverse path starts at dL over dL equals 1, moves left through all three operations using the saved activations, and branches to produce dL over d theta 3, dL over d theta 2, and dL over d theta 1 in the same sweep.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="18" y="22">FORWARD · COMPUTE ONCE AND SAVE WHAT LOCAL DERIVATIVES NEED</text>
			<rect class="viz-node viz-node--input" x="18" y="92" width="58" height="44" rx="5"></rect>
			<text class="viz-node-label" x="47" y="119">x</text>
			<rect class="viz-node viz-node--focus" x="122" y="92" width="58" height="44" rx="5"></rect>
			<text class="viz-node-label" x="151" y="111">f₁</text><text class="viz-node-value" x="151" y="127">uses θ₁</text>
			<rect class="viz-node viz-node--state" x="226" y="92" width="66" height="44" rx="12"></rect>
			<text class="viz-node-label" x="259" y="111">a₁</text><text class="viz-node-value" x="259" y="127">saved</text>
			<rect class="viz-node viz-node--focus" x="338" y="92" width="58" height="44" rx="5"></rect>
			<text class="viz-node-label" x="367" y="111">f₂</text><text class="viz-node-value" x="367" y="127">uses θ₂</text>
			<rect class="viz-node viz-node--state" x="442" y="92" width="66" height="44" rx="12"></rect>
			<text class="viz-node-label" x="475" y="111">a₂</text><text class="viz-node-value" x="475" y="127">saved</text>
			<rect class="viz-node viz-node--focus" x="554" y="92" width="58" height="44" rx="5"></rect>
			<text class="viz-node-label" x="583" y="111">f₃</text><text class="viz-node-value" x="583" y="127">uses θ₃</text>
			<rect class="viz-node viz-node--output" x="642" y="92" width="40" height="44" rx="20"></rect>
			<text class="viz-node-label" x="662" y="119">L</text>
			<path class="viz-forward" d="M76 114H121M180 114H225M292 114H337M396 114H441M508 114H553M612 114H641"></path>
			<text class="viz-axis-label" x="18" y="177">BACKWARD · ONE REVERSE SWEEP, NOT ONE SWEEP PER PARAMETER</text>
			<path class="viz-backward" d="M662 145V205H584"></path>
			<path class="viz-backward" d="M554 205H368"></path>
			<path class="viz-backward" d="M338 205H152"></path>
			<path class="viz-backward" d="M122 205H48"></path>
			<text class="viz-gradient-label" x="642" y="194">seed ∂L/∂L = 1</text>
			<text class="viz-gradient-label" x="475" y="194">VJP uses saved a₂</text>
			<text class="viz-gradient-label" x="259" y="194">VJP uses saved a₁</text>
			<path class="viz-backward" d="M583 205V247"></path>
			<path class="viz-backward" d="M367 205V247"></path>
			<path class="viz-backward" d="M151 205V247"></path>
			<rect class="viz-node viz-node--output" x="501" y="248" width="164" height="46" rx="5"></rect>
			<text class="viz-node-label" x="583" y="267">∂L/∂θ₃</text><text class="viz-node-value" x="583" y="284">accumulate at parameter</text>
			<rect class="viz-node viz-node--output" x="285" y="248" width="164" height="46" rx="5"></rect>
			<text class="viz-node-label" x="367" y="267">∂L/∂θ₂</text><text class="viz-node-value" x="367" y="284">accumulate at parameter</text>
			<rect class="viz-node viz-node--output" x="69" y="248" width="164" height="46" rx="5"></rect>
			<text class="viz-node-label" x="151" y="267">∂L/∂θ₁</text><text class="viz-node-value" x="151" y="284">accumulate at parameter</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow the solid arrows once to compute the scalar loss and save a₁ and a₂. Then start at ∂L/∂L = 1 and follow the dashed arrows left: each local vector-Jacobian product reuses a saved value and emits that operation's parameter gradient. The reverse path is traversed once, not once per θ. Original schematic checked against the <a href="https://jmlr.org/papers/v18/17-468.html">automatic-differentiation survey</a> and <a href="https://docs.pytorch.org/docs/stable/notes/autograd.html">PyTorch autograd mechanics</a>.</figcaption>
</figure>

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
