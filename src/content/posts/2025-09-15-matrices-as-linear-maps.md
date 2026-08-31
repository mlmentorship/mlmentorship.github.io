---
title: "Matrices as linear maps"
description: "A matrix is a linear function from one vector space to another. Every operation in ML. Projection, rotation, basis change, gradient flow. Is matrix multiplication."
date: "2025-09-15"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A matrix $A \in \mathbb{R}^{m \times n}$ represents a linear map $f: \mathbb{R}^n \to \mathbb{R}^m$ defined by $f(x) = Ax$. Composition of linear maps corresponds to matrix multiplication; the columns of $A$ are the images of the standard basis vectors.

Every layer in a neural network is a linear map (followed by a non-linearity). Every embedding lookup, every attention score, every gradient backward pass is a matrix multiplication. Understanding what a matrix *does* geometrically. Rather than just how to compute with it. Is the foundation for reasoning about model capacity, conditioning, and gradient flow.

## The geometry

For $A \in \mathbb{R}^{m \times n}$:

- **Columns of $A$** = images of $e_1, \dots, e_n$. Span them and you get the **column space** (range of the map).
- **Rows of $A$** = linear functionals; span the **row space**.
- **Null space** = $\{x : Ax = 0\}$. Directions the map collapses.
- **Rank** = dimension of column space = dimension of row space.

If $A$ is square and invertible, $A$ is a bijection: it stretches, rotates, and reflects $\mathbb{R}^n$ without losing information. If rank $< n$, $A$ collapses dimensions.

<!-- visual:matrix-columns-define-map -->
<figure class="learning-figure" aria-labelledby="matrix-columns-map-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="matrix-columns-map-title">Trace how the columns determine every output, including a direction that collapses to zero.</p>
	<div class="visual-panel">
		<svg viewBox="0 0 360 450" role="img" aria-labelledby="matrix-columns-svg-title matrix-columns-svg-desc">
			<title id="matrix-columns-svg-title">A rank-one matrix maps basis vectors to its columns and collapses a nonzero input</title>
			<desc id="matrix-columns-svg-desc">For A with rows one comma two and one comma two, the input basis vectors e one and e two map to columns a one equals one comma one and a two equals two comma two. The input x equals two e one minus e two is nonzero. In the output plane, both columns lie on the range line y equals x, and A x equals two a one minus a two equals zero. Numbered panels, coordinates, arrow directions, and line styles convey the result without relying on color.</desc>
			<defs>
				<marker id="matrix-columns-arrow-input" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
				<marker id="matrix-columns-arrow-focus" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="180" rx="4"></rect>
			<text class="viz-callout" x="24" y="36">1 · Read x in the input basis</text>
			<text class="viz-label" x="24" y="57">x = (2, −1) = 2e₁ − e₂</text>
			<path class="viz-axis" d="M42 132H318 M92 176V72"></path>
			<text class="viz-axis-label" x="300" y="150">e₁ axis</text>
			<text class="viz-axis-label" x="32" y="78">e₂ axis</text>
			<path class="viz-baseline" d="M92 132L146 132 M92 132L92 82"></path>
			<text class="viz-label" x="137" y="123">e₁</text>
			<text class="viz-label" x="101" y="87">e₂</text>
			<path class="viz-operating-guide" d="M92 132H218 M218 132V174"></path>
			<path class="viz-roc-curve" d="M92 132L218 174" marker-end="url(#matrix-columns-arrow-input)"></path>
			<text class="viz-callout" x="225" y="177">x ≠ 0</text>
			<rect class="viz-plot-bg" x="12" y="208" width="336" height="230" rx="4"></rect>
			<text class="viz-callout" x="24" y="232">2 · Use the same weights on A's columns</text>
			<text class="viz-label" x="24" y="253">A = [[1, 2], [1, 2]] · Ae₁ = a₁ · Ae₂ = a₂</text>
			<path class="viz-axis" d="M42 384H318 M84 420V272"></path>
			<path class="viz-baseline" d="M42 426L214 254"></path>
			<text class="viz-axis-label" x="218" y="271">range(A): y = x</text>
			<path class="viz-roc-curve" d="M84 384L126 342" marker-end="url(#matrix-columns-arrow-input)"></path>
			<text class="viz-label" x="132" y="344">a₁ = (1, 1)</text>
			<path class="viz-pr-curve" d="M80 380L168 292" marker-end="url(#matrix-columns-arrow-focus)"></path>
			<text class="viz-callout" x="174" y="294">a₂ = (2, 2) = 2a₁</text>
			<path class="viz-operating-guide" d="M174 298L88 384" marker-end="url(#matrix-columns-arrow-input)"></path>
			<text class="viz-label" x="178" y="318">then subtract a₂</text>
			<circle class="viz-operating-point" cx="84" cy="384" r="5"></circle>
			<text class="viz-callout" x="24" y="415">Ax = 2a₁ − a₂ = (0, 0)</text>
			<text class="viz-label" x="211" y="415">x is in null(A)</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> first map each input basis vector to its matching column: <var>Ae</var><sub>1</sub> = <var>a</var><sub>1</sub> and <var>Ae</var><sub>2</sub> = <var>a</var><sub>2</sub>. Then keep the input's coefficients. Here <var>x</var> = 2<var>e</var><sub>1</sub> − <var>e</var><sub>2</sub>, so <var>Ax</var> = 2<var>a</var><sub>1</sub> − <var>a</var><sub>2</sub> = 0. Both columns lie on one line, so the map has rank one and collapses a nonzero direction.</figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://math.mit.edu/~djk/18_022/chapter16/section01.html">MIT 18.022</a>, Peter Selinger's CC BY 4.0 <a href="https://www.mathstat.dal.ca/~selinger/linear-algebra/">Matrix Theory and Linear Algebra</a>, and Boyd and Vandenberghe's <a href="https://web.stanford.edu/~boyd/vmls/">Introduction to Applied Linear Algebra</a>.</p>

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
