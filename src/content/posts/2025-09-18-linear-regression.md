---
title: "Linear regression"
description: "Fit an affine prediction by squared residuals, verify its projection, and separate solver, sampling, penalty, and variance assumptions."
date: "2025-09-18"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Ordinary least squares chooses linear predictions that minimize squared residuals.
Its fitted vector is the orthogonal projection of observed responses onto the design matrix's column space.
This deterministic optimization problem does not require Gaussian noise; sampling assumptions justify additional statistical claims.

## Ordinary least squares (OLS)

Put observations in rows of $X$ and include a ones column if fitting an intercept.
With coefficient vector $\beta$, minimize $L=\|y-X\beta\|_2^2$.
Differentiating gives $\nabla_\beta L=-2X^\top(y-X\beta)$, so stationarity means $X^\top r=0$.

For full column rank, the unique coefficient vector is

$$
\widehat\beta=(X^\top X)^{-1}X^\top y.
$$

The fitted vector remains unique without full column rank, but coefficients need not be.
An SVD pseudoinverse selects the minimum-Euclidean-norm coefficient solution.

Take $x=(0,1,2)$ and $y=(1,2,2)$, with design rows $(1,x_i)$.
Then $X^\top X=\left[\begin{smallmatrix}3&3\\3&5\end{smallmatrix}\right]$ and $X^\top y=(5,6)^\top$.
Solving gives intercept $7/6$, slope $1/2$, and predictions $(7/6,5/3,13/6)$.
Residuals $(-1/6,1/3,-1/6)$ sum to zero, as does their dot product with $x$; SSE is $1/6$.

<!-- visual:ols-orthogonal-projection -->
<figure class="learning-figure" aria-labelledby="ols-projection-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ols-projection-title">The residual is perpendicular to the fitted column space</p><div class="visual-argument" data-visual-argument="ols-orthogonal-projection" data-argument-kind="geometry"><p class="visual-argument-premise">The residual is perpendicular to the fitted subspace</p><svg viewBox="0 0 360 457" role="img" aria-label="The residual is perpendicular to the fitted subspace. Projection of y=(3,2) onto the x-axis is (3,0). The residual (0,2) is orthogonal to that fitted column space, even if coefficient representations are nonunique."><title>The residual is perpendicular to the fitted subspace</title><desc>Projection of y=(3,2) onto the x-axis is (3,0). The residual (0,2) is orthogonal to that fitted column space, even if coefficient representations are nonunique.</desc><path d="M32 256.8L330 256.8" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M97.2 26L97.2 322" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="328" y="275.8" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="328" dy="0">x</tspan></text><text x="107.2" y="25" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="107.2" dy="0">y</tspan></text><path d="M97.2 256.8L262.8 146.4" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M258.2285237151706 154.15252246544583L262.8 146.4" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M253.88556530696178 147.63808485313263L262.8 146.4" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle cx="262.8" cy="146.4" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M97.2 256.8L262.8 256.8" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M254.6959760788259 260.7146898070011L262.8 256.8" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M254.6959760788259 252.88531019299893L262.8 256.8" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><circle cx="262.8" cy="256.8" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M97.2 256.8L97.2 146.4" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M101.11468980700107 154.5040239211741L97.2 146.4" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M93.28531019299894 154.5040239211741L97.2 146.4" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><circle cx="97.2" cy="146.4" r="3" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M262.8 256.8L262.8 146.4" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M20 349L45 349" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="57" y="353" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">Observed y: (3, 2)</tspan></text><path d="M20 381L45 381" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="57" y="385" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">Fitted projection: (3, 0)</tspan></text><path d="M20 413L45 413" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="57" y="417" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">Residual direction: (0, 2)</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 285" role="img" aria-labelledby="ols-projection-svg-title ols-projection-svg-desc">
			<title id="ols-projection-svg-title">Ordinary least squares as an orthogonal projection</title>
			<desc id="ols-projection-svg-desc">A slanted plane represents the column space of the design matrix X. From the origin, the fitted vector y-hat ends on that plane. The observed target vector y ends above it. A dashed residual vector connects y-hat to y and meets the plane at a marked right angle. This perpendicular residual satisfies X transpose r equals zero.</desc>
			<defs>
				<marker id="ols-projection-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
					<path d="M0 0L7 3.5L0 7Z" style="fill:var(--viz-edge)"></path>
				</marker>
				<marker id="ols-residual-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
					<path d="M0 0L7 3.5L0 7Z" style="fill:var(--viz-focus-stroke)"></path>
				</marker>
			</defs>
			<path d="M24 206L224 105L337 155L137 256Z" style="fill:var(--viz-state-bg);fill-opacity:.72;stroke:var(--viz-state-stroke);stroke-width:2"></path>
			<text class="viz-axis-label" x="340" y="183" text-anchor="end">Col(X)</text>
			<text class="viz-label" x="340" y="199" text-anchor="end">all possible fitted vectors Xw</text>
			<circle cx="67" cy="222" r="4" style="fill:var(--c-text);stroke:var(--c-text)"></circle>
			<text class="viz-label" x="45" y="241">origin</text>
			<path d="M67 222L207 148" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;marker-end:url(#ols-projection-arrow)"></path>
			<text class="viz-axis-label" x="111" y="169">fitted y_hat = Xw_hat</text>
			<path d="M67 222L251 55" style="fill:none;stroke:var(--c-text-soft);stroke-width:2.4;marker-end:url(#ols-projection-arrow)"></path>
			<text class="viz-axis-label" x="154" y="102">observed y</text>
			<path d="M207 148L251 55" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-dasharray:6 4;marker-end:url(#ols-residual-arrow)"></path>
			<text class="viz-callout" x="249" y="113">residual r</text>
			<text class="viz-label" x="249" y="127">y - y_hat</text>
			<path d="M207 148L219 154L225 142" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
			<circle cx="207" cy="148" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle>
			<circle cx="251" cy="55" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--c-text-soft);stroke-width:2"></circle>
			<rect x="194" y="223" width="142" height="42" rx="5" style="fill:var(--viz-neutral-bg);stroke:var(--c-rule);stroke-width:1.5"></rect>
			<text class="viz-axis-label" x="265" y="241" text-anchor="middle">r is perpendicular to Col(X)</text>
			<text class="viz-callout" x="265" y="257" text-anchor="middle">X^T r = 0</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> fitted responses lie in Col(X), an observation-space subspace. The closest fit leaves a perpendicular residual, giving X<sup>T</sup>r = 0. The picture is not a feature-scatter plot or advice to form an inverse. Use QR or SVD numerically. Original schematic follows <a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/positive-definite-matrices-and-applications/orthogonal-projections-and-their-applications/">MIT's projection notes</a> and the <a href="https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd141.htm">NIST least-squares handbook</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ols-orthogonal-projection"><strong>Check:</strong> Why is the residual orthogonal to X even when the coefficient vector is not unique?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ols-orthogonal-projection">Projection of y=(3,2) onto the x-axis is (3,0). The residual (0,2) is orthogonal to that fitted column space, even if coefficient representations are nonunique.</span></figcaption>
</figure>

The inverse expression identifies a solution; it is not the preferred numerical algorithm.

- **QR:** for full column rank, solve triangular $R\widehat\beta=Q^\top y$ without forming an inverse.
- **SVD:** handles rank deficiency with a declared numerical rank threshold.
- **Gradient methods:** useful when factorization costs exceed the available resources; stopping accuracy remains part of the contract.

## Probabilistic interpretation

Independent conditional Gaussian errors with a common variance make OLS a maximum-likelihood coefficient estimate.
Other error models can change efficiency, robustness, or interval calculations without changing the definition of least squares.

## Ridge and lasso

Regularization changes the objective rather than merely stabilizing an otherwise identical estimator.

| Method | Penalty | Effect |
|--------|---------|--------|
| **Ridge** (L2) | $\lambda\|w\|_2^2$ | Shrinks singular-direction coefficients; solve the modified linear system |
| **Lasso** (L1) | $\lambda\|w\|_1$ | Can produce exact zeros; use an appropriate convex solver |
| **Elastic net** | Both | Combines quadratic and absolute-value penalties |

### Penalize Only the Slope

**New task:** Add slope penalty $\lambda w^2$ with $\lambda=1$, leaving the intercept unpenalized.

**Picture and change:** Add $\operatorname{diag}(0,1)$ to the fixture's normal-equation matrix.

**Check:** Coefficients become $(4/3,1/3)$; the penalized slope no longer requires ordinary residual orthogonality.

**Use when:** A justified bias-variance trade-off favors shrinkage; coefficient sparsity alone does not establish causal importance.

### Weight by Precision

**New task:** Give the middle observation one-quarter the noise variance of the others.

**Picture and change:** Set $W=\operatorname{diag}(1,4,1)$ and solve $X^\top WX\widehat\beta=X^\top Wy$.

**Check:** Coefficients become $(4/3,1/2)$, satisfying $X^\top Wr=0$ rather than ordinary orthogonality.

**Use when:** Positive observation variances are known or defensibly estimated; the weighted inner product changes the projection.

## Assumptions and diagnostics

Separate conditional mean specification, error covariance, and distributional assumptions.
Residual plots against predictions/features can suggest misspecification or unequal variance; time plots can reveal dependence.
QQ plots assess distributional shape but do not prove exogeneity or independence.

## Gauss–Markov theorem

For full-column-rank fixed $X$, require $E[\varepsilon\mid X]=0$ and $\operatorname{Cov}(\varepsilon\mid X)=\sigma^2I$ with finite variance.
Then OLS has minimum conditional covariance among linear unbiased coefficient estimators: BLUE.
Normality is unnecessary for this theorem; it supports additional exact Gaussian-model inference.
Biased estimators can have smaller mean squared error.

<!-- learning:core-end -->
<div data-learning-extension>

## When to use vs. alternatives

- **Nonlinear mean:** justify basis functions, kernels, trees, or neural networks against held-out objectives.
- **Heavy tails/outliers:** compare robust losses; quantile regression targets a conditional quantile, not automatically the conditional mean.
- **Dependent observations:** consider covariance-aware estimation or hierarchical models and suitable inference.
- **Variable selection:** evaluate lasso or elastic net without treating selected features as automatically causal.

## Common pitfalls

- **Unidentified coefficients.** Duplicate columns permit offsetting coefficient changes without changing any fitted response.
- **Training-score claims.** Evaluate generalization with a split matching the sampling and deployment process.
- **Interval confusion.** Mean confidence intervals and individual prediction intervals target different quantities; the latter include observation noise.

## Reconstruct

<details>
<summary>If the observed vector is already in Col(X), what does the projection prove?</summary>

The fit has zero residual and zero training SSE.
This does not prove zero future error, correct sampling assumptions, or a causal coefficient interpretation.

</details>

</div>
