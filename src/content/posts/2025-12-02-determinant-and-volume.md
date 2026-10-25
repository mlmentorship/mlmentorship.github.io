---
title: "Determinant and volume"
description: "The determinant of a matrix is the signed volume scaling factor of the linear map. Zero determinant means the map collapses dimensions."
date: "2025-12-02"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A determinant gives the signed area or volume scale of a linear map.
Its magnitude measures the scale. Its sign records orientation.

Start with a unit square. Send its horizontal edge to $(2, 0.5)$ and its vertical edge to $(0.5, 1.25)$.
What area does the square occupy now? What happens if you exchange the two arrows?

## Geometric interpretation

Each column is the image of one basis vector. The two arrows bound the image of the square.
Follow their order in the first two panels. Then compare their span in the last panel.

<!-- visual:determinant-signed-area-collapse -->
<figure class="learning-figure" aria-labelledby="determinant-area-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="determinant-area-title">The same area can have either orientation. Parallel columns lose a dimension.</p>
	<div class="visual-panel">
		<svg viewBox="0 0 360 560" role="img" aria-labelledby="determinant-area-svg-title determinant-area-svg-desc">
			<title id="determinant-area-svg-title">Three ordered column pairs with positive, negative, and zero determinant</title>
			<desc id="determinant-area-svg-desc">The first panel shows columns a one equals two comma one half and a two equals one half comma one and one quarter, spanning a parallelogram of area two and one quarter in counterclockwise order, so the determinant is positive two and one quarter. The second panel uses the exact same parallelogram but swaps the column order: a one now points to one half comma one and one quarter and a two points to two comma one half. Its area remains two and one quarter, but the order is clockwise, so the determinant is negative two and one quarter. The third panel has columns b one equals two comma one and b two equals one comma one half. The second is half the first, so both arrows lie on one line, the parallelogram collapses to a segment, and the determinant is zero.</desc>
			<defs>
				<marker id="det-arrow-primary" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
				<marker id="det-arrow-secondary" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="164" rx="4"></rect>
			<text class="viz-callout" x="24" y="34">1 · Positive: area 2.25, orientation preserved</text>
			<path class="viz-gridline" d="M45 148H225 M65 160V48"></path>
			<polygon class="viz-node viz-node--focus" points="65,148 145,128 165,78 85,98"></polygon>
			<path d="M65 148L145 128" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 148L85 98" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="148" r="3"></circle>
			<text class="viz-label" x="139" y="146">a₁ = (2, 0.5)</text>
			<text class="viz-label" x="91" y="92">a₂ = (0.5, 1.25)</text>
			<text class="viz-callout" x="238" y="83">counterclockwise</text>
			<text class="viz-node-value" x="290" y="104">det[a₁ a₂]</text>
			<text class="viz-node-label" x="290" y="129">= +2.25</text>
			<text class="viz-label" x="238" y="154">density = 4/9</text>
			<rect class="viz-plot-bg" x="12" y="190" width="336" height="164" rx="4"></rect>
			<text class="viz-callout" x="24" y="212">2 · Swap columns: same area, opposite sign</text>
			<path class="viz-gridline" d="M45 326H225 M65 338V226"></path>
			<polygon class="viz-node viz-node--focus" points="65,326 145,306 165,256 85,276"></polygon>
			<path d="M65 326L85 276" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 326L145 306" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="326" r="3"></circle>
			<text class="viz-label" x="91" y="270">a₁ = (0.5, 1.25)</text>
			<text class="viz-label" x="139" y="324">a₂ = (2, 0.5)</text>
			<text class="viz-callout" x="238" y="261">clockwise</text>
			<text class="viz-node-value" x="290" y="282">det[a₁ a₂]</text>
			<text class="viz-node-label" x="290" y="307">= −2.25</text>
			<rect class="viz-plot-bg" x="12" y="368" width="336" height="180" rx="4"></rect>
			<text class="viz-callout" x="24" y="390">3 · Dependent columns: one direction is lost</text>
			<path class="viz-gridline" d="M45 522H225 M65 534V414"></path>
			<path d="M65 522L145 482" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 522L105 502" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="522" r="3"></circle>
			<text class="viz-label" x="139" y="476">b₁ = (2, 1)</text>
			<text class="viz-label" x="107" y="516">b₂ = (1, 0.5) = ½b₁</text>
			<text class="viz-callout" x="238" y="447">collapsed segment</text>
			<text class="viz-node-value" x="290" y="468">det[b₁ b₂]</text>
			<text class="viz-node-label" x="290" y="493">= 0</text>
			<text class="viz-label" x="238" y="519">area = 0</text>
			<text class="viz-label" x="238" y="536">not invertible</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> swapping the arrows changes their order but leaves the filled region unchanged. Only the determinant sign changes. In the last panel, both arrows lie on one line. There is no two-dimensional area.</figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-18-properties-of-determinants/">MIT OpenCourseWare 18.06</a>, the <a href="https://www.deeplearningbook.org/contents/linear_algebra.html">Deep Learning linear algebra chapter</a>, and the open <a href="https://math.libretexts.org/Bookshelves/Linear_Algebra/Interactive_Linear_Algebra_(Margalit_and_Rabinoff)/04%3A_Determinants/4.01%3A_Determinants-_Definition">Interactive Linear Algebra determinant text</a>.</p>

## Work the area

The columns give

$$
A=\begin{pmatrix}2&0.5\\0.5&1.25\end{pmatrix},\qquad
\det(A)=2(1.25)-0.5(0.5)=2.25.
$$

The four corners become $(0,0)$, $(2,0.5)$, $(0.5,1.25)$ and $(2.5,1.75)$.
The filled region has area $9/4$.

You can check that area by sliding each vertical slice downward by $x/4$.
The slide preserves every slice's height and width.
The arrows become $(2,0)$ and $(0.5,1.125)$.
The base is now horizontal: base $2$ times height $1.125$ gives $2.25$.

## Change of variables (probability)

Put uniform probability density $1$ on the original unit square.
The image has $9/4$ times the area, so its density must fall to $4/9$.
The total mass stays $(9/4)(4/9)=1$.

For this map, $Y=AX$ and

$$
A^{-1}=\frac49\begin{pmatrix}1.25&-0.5\\-0.5&2\end{pmatrix}.
$$

Thus $p_Y(y)=4/9$ inside the image and zero outside it.

More generally, let $Y=f(X)$ on a region where $f$ has a differentiable inverse.
Assume its Jacobian is nonsingular there. Then

$$
p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)|
$$

Here $J_{f^{-1}}$ is the inverse map's Jacobian.
The absolute value keeps probability density nonnegative, including when the map reverses orientation.

Normalizing flows use maps whose Jacobian determinants are cheap to compute.
For a triangular Jacobian, multiply its diagonal entries.
If several regular inverse branches reach the same output, add their density contributions.
The single-inverse formula alone does not handle that case.

## Change the map

### Exchange the columns

**New task:** keep the filled region but exchange the two basis images.

**Picture and change:** the second panel exchanges the arrows. The ordered area becomes $-2.25$.
Its magnitude remains $2.25$, so the transformed density remains $4/9$.

**Check:** a negative density would expose a missing absolute value.

**Use when:** tracking orientation or transporting probability mass requires the correct sign convention.

### Double every coordinate

**New task:** replace $A$ with $2A$.

**Picture and change:** both edge lengths double. In two dimensions, area grows by $2^2=4$.
Thus $\det(2A)=9$, and the uniform density becomes $1/9$.
In three dimensions, doubling every coordinate multiplies volume by $2^3=8$.

**Check:** doubling only one column multiplies the determinant by two.
Doubling the whole matrix multiplies it by $2^n$ in $n$ dimensions.

**Use when:** every coordinate unit changes, rather than one basis image alone.

## Properties

For square matrices $A,B\in\mathbb R^{n\times n}$:

- $\det(AB)=\det(A)\det(B)$. Composing maps multiplies their signed volume scales.
- $\det(A^\top)=\det(A)$.
- $\det(A^{-1})=1/\det(A)$ when $A$ is invertible.
- $\det(cA)=c^n\det(A)$.
- $\det(A)=\prod_i\lambda_i$, counting multiplicity. Real matrices can have complex eigenvalues.
- For a triangular matrix, $\det(A)=\prod_i A_{ii}$.
- For an orthogonal matrix $Q$, $|\det(Q)|=1$. Rotations and reflections preserve volume magnitude.

In exact arithmetic, these statements are equivalent:

$$
\det(A)=0
\iff \text{dependent columns}
\iff \text{not invertible}
\iff \text{a zero eigenvalue}.
$$

The last panel uses $B=\begin{pmatrix}2&1\\1&0.5\end{pmatrix}$.
Its second column is half its first. The image lies on a line, and $\det(B)=0$.
There is no ordinary two-dimensional inverse-density formula for this collapsed map.

## Try a different transformation

A reflection is followed by axis scalings of three and two.
Find the determinant and the image density of a unit-density square.

<details>
<summary>Check your answer</summary>

The determinant is $-6$. The image density is $1/6$ on its support.
The reflection changes orientation. The scale magnitudes multiply the area by six.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Computing determinants

| Method | Cost | When |
|--------|------|------|
| LU decomposition | $O(n^3)$ | General-purpose; standard library default |
| Triangular: product of diag | $O(n)$ | When matrix is already triangular |
| Eigendecomposition: $\prod \lambda_i$ | $O(n^3)$ | If you need eigenvalues anyway |
| **Log-determinant** for PD matrices | via Cholesky | When you only need $\log \det$ (e.g., Gaussian log-likelihood) |

For a general matrix, use a signed log-determinant.
Track row-pivot signs and the signs of the LU diagonal entries. Sum $\log|U_{ii}|$ for the log-magnitude.
For a positive-definite matrix with $A=LL^\top$, compute $\log\det(A)=2\sum_i\log L_{ii}$.
An explicit determinant can overflow or underflow.

## Common pitfalls

- **Using $\det$ as a proxy for matrix "size."** A nearly singular matrix can have huge entries but tiny $\det$.
- **Computing $\det$ for invertibility tests.** Numerically unstable; use rank or condition number instead.
- **Forgetting the absolute value in change-of-variables formulas.** Densities are non-negative; the Jacobian determinant can be negative.

</div>
