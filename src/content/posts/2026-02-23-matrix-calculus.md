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

<!-- visual:jacobian-transpose-pullback -->
<figure class="learning-figure plot-panel" aria-labelledby="jacobian-pullback-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="jacobian-pullback-title">Use shapes to see how the transposed Jacobian pulls two output gradients back to three parameters.</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="jacobian-pullback-svg-title jacobian-pullback-svg-desc">
		<title id="jacobian-pullback-svg-title">A numeric vector-to-vector chain rule using a transposed Jacobian</title>
		<desc id="jacobian-pullback-svg-desc">A function maps three parameters to two outputs, so its Jacobian has two rows and three columns. The upstream loss gradient is the two-vector 5, negative 2. Transposing the Jacobian produces a three-by-two matrix. Multiplication gives the three parameter gradients 8, negative 11, and negative 8. Each parameter gradient combines that parameter's sensitivity through both outputs.</desc>
		<defs><marker id="jacobian-pullback-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<text class="viz-axis-label" x="18" y="22">FORWARD SHAPE: 3 PARAMETERS -&gt; 2 OUTPUTS</text>
		<rect class="viz-node viz-node--input" x="18" y="38" width="115" height="52" rx="4"></rect>
		<text class="viz-label" x="75.5" y="58" text-anchor="middle">theta</text>
		<text class="viz-callout" x="75.5" y="78" text-anchor="middle">3 x 1</text>
		<path d="M137 64H217" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#jacobian-pullback-arrow)"></path>
		<text class="viz-label" x="177" y="53" text-anchor="middle">f, with J: 2 x 3</text>
		<rect class="viz-node viz-node--output" x="221" y="38" width="121" height="52" rx="4"></rect>
		<text class="viz-label" x="281.5" y="58" text-anchor="middle">y = f(theta)</text>
		<text class="viz-callout" x="281.5" y="78" text-anchor="middle">2 x 1</text>
		<text class="viz-axis-label" x="18" y="122">BACKWARD SHAPE: J^T g_y = g_theta</text>
		<rect class="viz-node viz-node--focus" x="18" y="138" width="165" height="116" rx="4"></rect>
		<text class="viz-label" x="100.5" y="158" text-anchor="middle">J^T: 3 x 2</text>
		<text class="viz-callout" x="100.5" y="184" text-anchor="middle">[  2    1 ]</text>
		<text class="viz-callout" x="100.5" y="207" text-anchor="middle">[ -1    3 ]</text>
		<text class="viz-callout" x="100.5" y="230" text-anchor="middle">[  0    4 ]</text>
		<text class="viz-callout" x="195" y="199" text-anchor="middle">x</text>
		<rect class="viz-node viz-node--input" x="207" y="138" width="62" height="116" rx="4"></rect>
		<text class="viz-label" x="238" y="158" text-anchor="middle">g_y</text>
		<text class="viz-callout" x="238" y="194" text-anchor="middle">[  5 ]</text>
		<text class="viz-callout" x="238" y="222" text-anchor="middle">[ -2 ]</text>
		<text class="viz-callout" x="280" y="199" text-anchor="middle">=</text>
		<rect class="viz-node viz-node--output" x="291" y="138" width="51" height="116" rx="4"></rect>
		<text class="viz-label" x="316.5" y="158" text-anchor="middle">g_theta</text>
		<text class="viz-callout" x="316.5" y="184" text-anchor="middle">[ 8 ]</text>
		<text class="viz-callout" x="316.5" y="207" text-anchor="middle">[-11]</text>
		<text class="viz-callout" x="316.5" y="230" text-anchor="middle">[-8 ]</text>
		<path d="M180 262V280" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#jacobian-pullback-arrow)"></path>
		<text class="viz-axis-label" x="18" y="300">EACH ROW COLLECTS BOTH OUTPUT ROUTES</text>
		<rect class="viz-node" x="18" y="314" width="324" height="96" rx="4"></rect>
		<text class="viz-callout" x="30" y="339">dL/dtheta1 =  2(5) + 1(-2) =   8</text>
		<text class="viz-callout" x="30" y="366">dL/dtheta2 = -1(5) + 3(-2) = -11</text>
		<text class="viz-callout" x="30" y="393">dL/dtheta3 =  0(5) + 4(-2) =  -8</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> rows of <em>J</em> belong to outputs and columns belong to parameters. Transposing <em>J</em> puts one parameter on each result row, so multiplying by the two upstream loss gradients sums every output route into that parameter's gradient. The inner dimensions cancel: <code>(3 x 2)(2 x 1) = 3 x 1</code>. This is an original worked example checked against the <a href="https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html">JAX vector-Jacobian product contract</a>.</figcaption>
</figure>

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
