---
title: "Factorization machines"
description: "Share vectors across feature pairs, derive sparse interaction evaluation, and separate available unseen-pair estimates from validated generalization."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Suppose training contains feature pairs A-B and B-C, but not A-C.
A factorization machine assigns a shared vector to each feature, making the A-C dot product computable.
That supplies an estimate, not evidence that the estimate is accurate.

For $d$ features, give feature $i$ a scalar weight $w_i$ and vector $v_i\in\mathbb R^k$.
With global bias $w_0$, the prediction score is

$$
\hat{y}(x) = w_0 + \sum_i w_i x_i + \sum_{i < j} \langle v_i, v_j \rangle x_i x_j.
$$

An independent degree-two model stores $d(d-1)/2$ pair coefficients; FM stores $dk$ interaction parameters plus main effects.
This changes statistical sharing as well as storage when $k\ll d$.
An absent pair's independent coefficient lacks direct co-occurrence evidence; shared vectors impose an additional modeling assumption.
The following vectors are chosen fixtures, not fitted outputs.

<!-- visual:fm-shared-embeddings-unseen-pair -->
<figure class="learning-figure plot-panel" aria-labelledby="fm-sharing-visual-title">
	<p class="visual-kicker">Parameter sharing</p>
	<p class="visual-title" id="fm-sharing-visual-title">Shared embeddings estimate an unseen pair without a dedicated pair weight</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 430" role="img" aria-labelledby="fm-sharing-svg-title fm-sharing-svg-desc">
			<title id="fm-sharing-svg-title">Pair-specific weights compared with shared feature embeddings for an unseen interaction</title>
			<desc id="fm-sharing-svg-desc">An original three-feature example compares two parameterizations. In a polynomial model, training included pairs A-B and B-C, so their separate edge weights are learned, while the never-observed A-C edge has an unknown pair-specific weight. In a factorization machine, each feature instead owns one reusable two-dimensional vector: A is one-zero, B is point-eight-point-two, and C is point-six-point-four. The observed A-B and B-C pairs help learn those vectors. Even though A-C never appeared together, its coefficient is defined and computable as the dot product of A and C, one times point six plus zero times point four, which equals point six. This demonstrates availability of an estimate, not guaranteed accuracy on every unseen pair.</desc>
			<text class="viz-axis-label" x="14" y="20">PAIR-SPECIFIC MODEL · EDGE PARAMETERS</text>
			<rect class="viz-node viz-node--input" x="18" y="42" width="76" height="38" rx="4"></rect>
			<text class="viz-callout" x="56" y="65" text-anchor="middle">feature A</text>
			<rect class="viz-node viz-node--input" x="142" y="42" width="76" height="38" rx="4"></rect>
			<text class="viz-callout" x="180" y="65" text-anchor="middle">feature B</text>
			<rect class="viz-node viz-node--input" x="266" y="42" width="76" height="38" rx="4"></rect>
			<text class="viz-callout" x="304" y="65" text-anchor="middle">feature C</text>
			<path class="viz-pr-curve" d="M94 61H142 M218 61H266"></path>
			<text class="viz-label" x="118" y="51" text-anchor="middle">wAB</text>
			<text class="viz-label" x="242" y="51" text-anchor="middle">wBC</text>
			<path class="viz-baseline" d="M56 84C56 132 304 132 304 84"></path>
			<rect class="viz-node viz-node--focus" x="107" y="102" width="146" height="42" rx="4"></rect>
			<text class="viz-callout" x="180" y="120" text-anchor="middle">wAC = ?</text>
			<text class="viz-label" x="180" y="136" text-anchor="middle">A–C never appeared together</text>
			<path class="viz-gridline" d="M8 164H352"></path>
			<text class="viz-axis-label" x="14" y="188">FACTORIZATION MACHINE · PARAMETERS LIVE ON FEATURES</text>
			<rect class="viz-node viz-node--input" x="18" y="207" width="96" height="62" rx="4"></rect>
			<text class="viz-callout" x="66" y="228" text-anchor="middle">feature A</text>
			<text class="viz-label" x="66" y="248" text-anchor="middle">vA = [1, 0]</text>
			<text class="viz-label" x="66" y="262" text-anchor="middle">learned via A–B</text>
			<rect class="viz-node viz-node--input" x="132" y="207" width="96" height="62" rx="4"></rect>
			<text class="viz-callout" x="180" y="228" text-anchor="middle">feature B</text>
			<text class="viz-label" x="180" y="248" text-anchor="middle">vB = [.8, .2]</text>
			<text class="viz-label" x="180" y="262" text-anchor="middle">shared by both</text>
			<rect class="viz-node viz-node--input" x="246" y="207" width="96" height="62" rx="4"></rect>
			<text class="viz-callout" x="294" y="228" text-anchor="middle">feature C</text>
			<text class="viz-label" x="294" y="248" text-anchor="middle">vC = [.6, .4]</text>
			<text class="viz-label" x="294" y="262" text-anchor="middle">learned via B–C</text>
			<path class="viz-pr-curve" d="M114 226H132 M228 226H246"></path>
			<text class="viz-label" x="123" y="216" text-anchor="middle">AB</text>
			<text class="viz-label" x="237" y="216" text-anchor="middle">BC</text>
			<path class="viz-baseline" d="M66 274C66 290 72 298 85 302 M275 302C288 298 294 290 294 274"></path>
			<text class="viz-callout" x="180" y="309" text-anchor="middle">unseen pair · coefficient still defined</text>
			<rect class="viz-node viz-node--focus" x="32" y="326" width="296" height="72" rx="4"></rect>
			<text class="viz-callout" x="180" y="348" text-anchor="middle">wAC := ⟨vA, vC⟩</text>
			<text class="viz-callout" x="180" y="369" text-anchor="middle">= 1 × .6 + 0 × .4 = .60</text>
			<text class="viz-label" x="180" y="388" text-anchor="middle">an available estimate, not a guarantee of accuracy</text>
			<text class="viz-axis-label" x="180" y="424" text-anchor="middle">O(d²) EDGE WEIGHTS → O(dk) FEATURE EMBEDDINGS</text>
		</svg>
	</div>
	<table aria-label="Aggregate interaction check with all three feature values equal to one">
		<tbody>
			<tr><th scope="row">Explicit AB + AC + BC</th><td>0.80 + 0.60 + 0.56 = 1.96</td></tr>
			<tr><th scope="row">Squared norm of summed vectors</th><td>2.4² + 0.6² = 6.12</td></tr>
			<tr><th scope="row">Sum of individual squared norms</th><td>1 + 0.68 + 0.52 = 2.20</td></tr>
			<tr><th scope="row">Half the difference</th><td>(6.12 − 2.20) / 2 = 1.96</td></tr>
		</tbody>
	</table>
	<figcaption><strong>Read it this way:</strong> one reusable vector per feature supplies the unseen A-C coefficient 0.60. With all inputs active, explicit pair summation and the aggregate identity below both give 1.96. Neither calculation validates unseen-pair accuracy. Original example based on <a href="https://doi.org/10.1109/ICDM.2010.127">Rendle (2010)</a>. <span class="visual-reader-check" data-reader-check="fm-shared-embeddings-unseen-pair"><strong>Check:</strong> Which learned objects supply an A-C estimate when there is no dedicated A-C parameter?</span></figcaption>
</figure>

## The mechanism

For each coordinate $f$, expanding $(\sum_i v_{i,f}x_i)^2$ gives diagonal terms plus twice every $i<j$ pair.
Subtract those self-terms, divide by two, and sum over coordinates:

$$
\sum_{i < j} \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^{k} \left( \left(\sum_i v_{i,f} x_i\right)^2 - \sum_i v_{i,f}^2 x_i^2 \right).
$$

For $s=\operatorname{nnz}(x)$ active inputs, this costs $O(ks)$ instead of enumerating $O(ks^2)$ vector pairs.
Parameter storage still depends on the full feature vocabulary, $O(dk)$.

Using the illustrated vectors and $x=(1,1,1)$, pair coefficients are $0.8,0.6,0.56$, summing to $1.96$.
The vector sum is $(2.4,0.6)$; half of $6.12-2.20$ also gives $1.96$.
Deactivating B leaves only A-C and score $0.6$, with bias and linear terms zero.

## Sparse one-hot inputs

Categorical IDs can be one-hot encoded; numeric inputs can also participate with their actual values.
A pair contributes only when both input values are nonzero.

### Change active inputs

**New task:** activate only one user indicator and one item indicator, both equal to one.

**Picture and change:** remove every inactive linear and interaction term:

$$
\hat{y} = w_0 + w_{\text{user}} + w_{\text{item}} + \langle v_{\text{user}}, v_{\text{item}} \rangle.
$$

**Check:** fitted values $w_0=3,w_{\rm user}=0.2,w_{\rm item}=-0.1$ and vectors $(2,1),(1.5,1)$ give $7.1$.
This is biased matrix factorization's score, not an automatically bounded rating or probability.
Adding category, time, or device features introduces their linear terms and active pairs, not explicit third-order terms.

**Use when:** compare FM with MF on matched features.

### Change field specificity

**New task:** let a category feature interact differently with user and time fields.

**Picture and change:** field-aware FM uses $\langle v_{i,F(j)},v_{j,F(i)}\rangle$ for feature fields $F$.
In a separate one-dimensional fixture, category A uses value one against users and two against time.
The counterpart vectors are three and four, respectively.

**Check:** category-user interaction is $1(3)=3$; category-time is $2(4)=8$.
Reusing A's value one for both would give four for the latter pair.
The plain-FM aggregate-square computation cannot simply reuse one vector per feature here.

**Use when:** measured field-specific benefits justify added storage and compute.

<!-- learning:core-end -->

<div data-learning-extension>

## Variants

- [Field-aware FM](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) has field-indexed feature vectors, increasing storage to roughly $O(dkF)$ for $F$ fields.
- [DeepFM](https://arxiv.org/abs/1703.04247) combines an FM component with a deep branch sharing feature embeddings.
- xDeepFM uses a compressed interaction network; AutoInt uses attention; DCN uses cross layers. Their interaction computations differ.

## Tradeoffs

FM shares pair coefficients instead of fitting them independently, trading flexibility for statistical reuse.
Compare logistic regression, boosting, and interaction models with matched features and task metrics.
Architectural complexity does not guarantee better accuracy, speed, or interpretation.
Choose the loss/link separately: binary outcomes might use a sigmoid and logistic loss on the score.

## Common pitfalls

Keep main effects: with one active feature, there is no pair term.
Numeric-feature scaling changes interaction magnitudes; normalize or otherwise validate preprocessing rather than requiring arbitrary binning.
Tune $k$ and regularization, especially for rare features.
A new untrained feature needs an unknown-feature/initialization policy; an available unseen-pair coefficient does not solve unseen-feature support.
Removing Gram diagonal terms does not imply the remaining off-diagonal matrix itself has rank $k$.

Does plain FM explicitly model arbitrary third-order interactions?

<details><summary>Answer</summary>

No. Its explicit interaction polynomial has degree two; higher-order behavior requires another computation.

</details>

Original formulation: [Rendle (2010)](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf).

</div>

## Related

- [Matrix factorization for recsys](/concepts/matrix-factorization-recsys/).
- [Alternating least squares](/concepts/alternating-least-squares/).
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity/).
