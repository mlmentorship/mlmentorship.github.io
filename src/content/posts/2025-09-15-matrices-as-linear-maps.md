---
title: "Matrices as linear maps"
description: "A matrix sends input basis vectors to its columns. Weight those columns to find any output and identify directions that collapse."
date: "2025-09-15"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A matrix describes a linear map. Its columns say where the input basis vectors land.
The same weights that build an input also combine its column images.

Consider $A=\begin{pmatrix}1&2\\1&2\end{pmatrix}$ and the nonzero input $x=(2,-1)$.
Can its output be zero? Can we predict that output from the two column arrows?

## The geometry

Let $e_1=(1,0)$ and $e_2=(0,1)$ be the input basis vectors.
Their images are columns $a_1=(1,1)$ and $a_2=(2,2)$.
The second column is twice the first. Both point along one output line.

<!-- visual:matrix-columns-define-map -->
<figure class="learning-figure" aria-labelledby="matrix-columns-map-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="matrix-columns-map-title">A nonzero input can collapse to zero when matrix columns are dependent</p>
	<div class="visual-panel">
		<svg viewBox="0 0 360 450" role="img" aria-labelledby="matrix-columns-svg-title matrix-columns-svg-desc">
			<title id="matrix-columns-svg-title">A rank-one matrix maps basis vectors to its columns and collapses a nonzero input</title>
			<desc id="matrix-columns-svg-desc">The matrix has rows (1, 2) and (1, 2). Basis vectors map to columns a1 = (1, 1) and a2 = (2, 2). The nonzero input x = (2, -1) equals twice e1 minus e2. Its output is twice a1 minus a2, which is zero. Both columns lie on the output line y = x.</desc>
			<defs>
				<marker id="matrix-columns-arrow-input" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
				<marker id="matrix-columns-arrow-focus" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="180" rx="4"></rect>
			<text class="viz-callout" x="24" y="36">1 · Read x in the input basis</text>
			<text class="viz-label" x="24" y="57">x = (2, −1) = 2e₁ − e₂</text>
			<path class="viz-axis" d="M42 132H318 M92 186V72"></path>
			<text class="viz-axis-label" x="300" y="150">e₁ axis</text>
			<text class="viz-axis-label" x="32" y="78">e₂ axis</text>
			<path class="viz-baseline" d="M92 132L146 132 M92 132L92 82"></path>
			<text class="viz-label" x="137" y="123">e₁</text>
			<text class="viz-label" x="101" y="87">e₂</text>
			<path class="viz-operating-guide" d="M92 132H200 M200 132V182"></path>
			<path class="viz-roc-curve" d="M92 132L200 182" marker-end="url(#matrix-columns-arrow-input)"></path>
			<text class="viz-callout" x="207" y="177">x ≠ 0</text>
			<rect class="viz-plot-bg" x="12" y="208" width="336" height="230" rx="4"></rect>
			<text class="viz-callout" x="24" y="232">2 · Use the same weights on A's columns</text>
			<text class="viz-label" x="24" y="253">A = [[1, 2], [1, 2]] · Ae₁ = a₁ · Ae₂ = a₂</text>
			<path class="viz-axis" d="M42 384H318 M84 420V272"></path>
			<path class="viz-baseline" d="M42 426L214 254"></path>
			<text class="viz-axis-label" x="218" y="271">range(A): y = x</text>
			<path class="viz-pr-curve" d="M84 384L168 300" marker-end="url(#matrix-columns-arrow-focus)"></path>
			<path class="viz-roc-curve" d="M84 384L126 342" marker-end="url(#matrix-columns-arrow-input)"></path>
			<text class="viz-label" x="132" y="344">a₁ = (1, 1)</text>
			<text class="viz-callout" x="174" y="294">a₂ = (2, 2) = 2a₁</text>
			<path class="viz-operating-guide" d="M168 300L84 384" marker-end="url(#matrix-columns-arrow-input)"></path>
			<text class="viz-label" x="178" y="318">then subtract a₂</text>
			<circle class="viz-operating-point" cx="84" cy="384" r="5"></circle>
			<text class="viz-callout" x="24" y="415">Ax = 2a₁ − a₂ = (0, 0)</text>
			<text class="viz-label" x="211" y="415">x is in null(A)</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> use the input's weights on the column arrows. Twice the first column equals the second, so subtraction reaches zero. The input was nonzero; this map loses one direction. <span class="visual-reader-check" data-reader-check="matrix-columns-define-map"><strong>Check:</strong> How can x be nonzero while 2a1-a2 is zero? What does that reveal about A?</span></figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://math.mit.edu/~djk/18_022/chapter16/section01.html">MIT 18.022</a>, Peter Selinger's CC BY 4.0 <a href="https://www.mathstat.dal.ca/~selinger/linear-algebra/">Matrix Theory and Linear Algebra</a>, and Boyd and Vandenberghe's <a href="https://web.stanford.edu/~boyd/vmls/">Introduction to Applied Linear Algebra</a>.</p>

## Work one output

Write $x=2e_1-e_2$. Linearity gives

$$
Ax=2Ae_1-Ae_2=2a_1-a_2=2(1,1)-(2,2)=(0,0).
$$

For any input $(x_1,x_2)$, the output is $(x_1+2x_2)(1,1)$.
Thus the range is the line spanned by $(1,1)$.
The null space contains every multiple of $(2,-1)$.
Check $(1,1)$: column weighting and ordinary multiplication both give $(3,3)$.

## General form and shapes

With $n$ input coordinates and $m$ output coordinates, $A\in\mathbb R^{m\times n}$ and $f(x)=Ax$.
Linearity means $f(\alpha x+\beta z)=\alpha f(x)+\beta f(z)$ for inputs $x,z$ and scalar weights $\alpha,\beta$.
Since $x=\sum_{j=1}^n x_j e_j$, we get $Ax=\sum_{j=1}^n x_j a_j$.

- The **column space**, or range, contains all outputs. It lies in $\mathbb R^m$.
- Each row computes one output coordinate as a linear functional. The **row space** lies in $\mathbb R^n$.
- The **null space**, or kernel, is $\{x:Ax=0\}$. These input directions disappear.
- **Rank** is the column-space dimension, equal to the row-space dimension.

Rank plus null-space dimension equals $n$. Here $1+1=2$.
Independent columns prevent input information loss. A square matrix with independent columns is also onto and invertible.

## Change the map

### 1. Produce three coordinates

**New task:** send the same two input coordinates to a three-coordinate output.

**Picture and change:** choose columns $(1,0,1)$ and $(0,1,1)$. Keep the input weights $2,-1$.

**Check:** $2(1,0,1)-(0,1,1)=(2,-1,1)$.
The columns are independent, so no nonzero input collapses. The map is not onto all of $\mathbb R^3$.

**Use when:** a layer changes feature count. Inputs count columns; outputs count rows.

### 2. Add an output offset

**New task:** compute $f(x)=Ax+b$ with $b=(1,0)$.

**Picture and change:** combine columns as before, then translate the result by $b$.
The complete map is affine, not linear.

**Check:** $f(2,-1)=(1,0)$ and $f(0)=b\ne0$.
Its values at basis vectors are no longer simply the columns of $A$.

**Use when:** a fixed bias displaces the origin.

## Composition and multiplication

For $A$ of shape $m\times n$ and $B$ of shape $k\times m$, applying $A$ then $B$ gives $BA$.
The result has shape $k\times n$.
Associativity follows from composition: $C(BA)x=(CB)Ax$.
Reversing the maps generally changes the result.

## A projection need not be perpendicular

Let $P=\begin{pmatrix}1&1\\0&0\end{pmatrix}$. It satisfies $P^2=P$, but sends $(0,1)$ to $(1,0)$.
Its kernel is the line spanned by $(-1,1)$, not the perpendicular complement of its range.
For an orthogonal projection, also require $P^\top=P$.

## Try reversing two maps

Let $A=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ and $B=\operatorname{diag}(2,1)$.
Compare $BAe_2$ with $ABe_2$.

<details>
<summary>Check your answer</summary>

$Ae_2=(1,1)$, so $BAe_2=(2,1)$.
But $Be_2=(0,1)$, so $ABe_2=(1,1)$.
The order changes the output.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Special families

| Matrix | Geometric action |
|--------|----------------|
| Orthogonal $Q$ ($Q^\top Q = I$) | Rotation or reflection (preserves length and angle) |
| Diagonal | Independent scaling along each axis |
| Symmetric | Has real eigenvalues; orthogonal eigenvector basis |
| Positive definite | Symmetric + all eigenvalues > 0; defines an inner product |
| Permutation | Reorders coordinates |
| Projection ($P^2 = P$) | Maps onto its range along its kernel; orthogonal only when also symmetric |

## Common pitfalls

- **Treating matrix multiplication as element-wise.** Use Hadamard ($\odot$) for element-wise; matrix multiplication is composition.
- **Forgetting that shapes determine the map.** $A \in \mathbb{R}^{3 \times 5}$ is a map $\mathbb{R}^5 \to \mathbb{R}^3$, not the other way around.
- **Confusing column space with row space.** Both have dimension = rank, but they live in different spaces ($\mathbb{R}^m$ vs $\mathbb{R}^n$).

A fixed-weight dense operation $Ax$ is linear. Adding bias makes it affine; applying a nonlinear activation changes the map again.
Neural networks and input-dependent attention are not generally linear functions of their inputs.

</div>
