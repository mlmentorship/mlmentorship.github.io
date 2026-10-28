---
title: "Matrix calculus for ML"
description: "Gradients, Jacobians, and Hessians for vector- and matrix-valued functions. The minimum needed to derive backprop and second-order methods."
date: "2026-02-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Matrix calculus transports local changes through vector-valued maps while preserving which output depends on which parameter.
Use column gradients and the Jacobian convention $J_{ij}=\partial y_i/\partial\theta_j$ throughout this lesson.
For a scalar loss, the gradient has the parameter's shape; the Jacobian records every output-parameter sensitivity.
Twice continuously differentiable scalar losses also have a symmetric Hessian of second derivatives.

## The four shape combinations

| $f: \mathbb{R}^n \to \mathbb{R}$ | $f: \mathbb{R}^n \to \mathbb{R}^m$ |
|---|---|
| **Gradient** $\nabla f \in \mathbb{R}^n$ | **Jacobian** $J_f \in \mathbb{R}^{m \times n}$, $(J_f)_{ij} = \partial f_i / \partial x_j$ |
| Second derivative: **Hessian** $\nabla^2 f \in \mathbb{R}^{n \times n}$, $(\nabla^2 f)_{ij} = \partial^2 f / \partial x_i \partial x_j$ | (rarely used; tensor-valued) |

Layout names differ across references; inspect the declared Jacobian entries and dimensions rather than guessing from a field's terminology.
Define the matrix gradient by $dL=\operatorname{tr}((\nabla_W L)^\top dW)$.
For $W\in\mathbb R^{m\times n}$, that gradient has shape $m\times n$.

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

Let $y=f(\theta)\in\mathbb R^m$ and let $g_y=\nabla_y L$ be the incoming loss gradient.
Since $dy=J_f\,d\theta$, the loss change is

$$
dL=g_y^\top dy=g_y^\top J_f\,d\theta=(J_f^\top g_y)^\top d\theta.
$$

Therefore $\nabla_\theta L=J_f^\top g_y$.
This reverse operation is a vector-Jacobian product under the usual row-cotangent terminology, not the forward product $J_fv$.

For $f(\theta)=(2\theta_1-\theta_2,\ \theta_1+3\theta_2+4\theta_3)^\top$, the Jacobian is

$$
J_f=\begin{bmatrix}2&-1&0\\1&3&4\end{bmatrix}.
$$

With $g_y=(5,-2)^\top$, its transpose returns $(8,-11,-8)^\top$.
For example, parameter two receives $(-1)(5)+(3)(-2)=-11$ through the two output routes.

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
	<figcaption><strong>Read it this way:</strong> rows of <em>J</em> belong to outputs; columns belong to parameters. Transposition collects every output route into each parameter's gradient. The shapes are <code>(3 x 2)(2 x 1) = 3 x 1</code>. The example follows the <a href="https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html">JAX vector-Jacobian product contract</a>. <span class="visual-reader-check" data-reader-check="jacobian-transpose-pullback"><strong>Check:</strong> Why does theta2 receive -1 times 5 plus 3 times -2 rather than only one output's contribution?</span></figcaption>
</figure>

## Change the Requirement

### Differentiate a Trainable Matrix

**New task:** Differentiate $y=Wx+b$ with respect to the matrix, input, and bias.

**Picture and change:** Expand $dy=(dW)x+W\,dx+db$ inside $dL=g_y^\top dy$.

$$
\nabla_W L=g_yx^\top,\qquad \nabla_xL=W^\top g_y,\qquad \nabla_bL=g_y.
$$

**Check:** For $x=(1,2)^\top$ and $g_y=(5,-2)^\top$, $\nabla_WL=\begin{bmatrix}5&10\\-2&-4\end{bmatrix}$.

**Use when:** Parameters are arrays rather than one vector.
For a shared bias across examples, sum example gradients; a mean loss additionally divides by batch size.

## Hessian and second-order methods

**New task:** Obtain curvature along a fixed direction $v$ without storing the Hessian $H$.

**Picture and change:** Differentiate the scalar $\nabla L(\theta)^\top v$ with respect to $\theta$.
For a twice continuously differentiable loss, this returns $Hv$.

**Check:** $L(\theta)=\theta_1^2+3\theta_2^2$ has $H=\operatorname{diag}(2,6)$.
For $v=(1,-1)^\top$, the result is $(2,-6)^\top$.
Its Newton step reaches the minimum in one update because this particular loss is a strictly convex quadratic.

**Use when:** A solver needs Hessian products, not all $P^2$ entries.
Automatic differentiation can compose forward and reverse operations without materializing $H$.
An indefinite Hessian does not guarantee a descending Newton direction; damping or trust-region safeguards may be needed.

Adam's second moment is an average of squared stochastic gradients, not generally a Hessian diagonal.
K-FAC approximates Fisher-related blocks; Shampoo uses gradient-based preconditioners.
Neither should be described as an exact Hessian inverse.

<!-- learning:core-end -->
<div data-learning-extension>

## Common pitfalls

- **Wrong orientation.** Multiplying a $2\times3$ Jacobian by a two-coordinate incoming gradient is dimensionally invalid.
	Its transpose has the required $3\times2$ shape.
- **Wrong parameter shape.** A scalar loss can have a matrix gradient.
- **Full Jacobian allocation.** A dense $m\times n$ Jacobian has $mn$ entries.
	Reverse products avoid building that matrix when only one incoming gradient is needed.
- **Wrong reduction.** Sum shared-parameter contributions before applying the loss's declared sum or mean normalization.

## Reconstruct

Why is the gradient of $x^\top A x$ not always $2Ax$?

<details>
<summary>Reasoned answer</summary>

The differential contributes through both occurrences of $x$, giving $(A+A^\top)x$.
It reduces to $2Ax$ only when $A$ is symmetric.
For $A=\begin{bmatrix}0&1\\0&0\end{bmatrix}$ and $x=(1,2)^\top$, the gradient is $(2,1)^\top$, not $(4,0)^\top$.

</details>

</div>
