---
title: "SVD and PCA"
description: "Separate singular directions of a linear map from principal directions of centered data, then choose rank using reconstruction error."
date: "2025-09-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

SVD identifies input directions, output directions, and nonnegative scaling factors for any real matrix.
PCA applies that decomposition to centered observations, retaining directions that minimize Euclidean squared reconstruction error.
Large retained variance does not guarantee retained label information.

## The decomposition

$A\in\mathbb R^{m\times n}$ has $A=U\Sigma V^\top$.
Full factors have orthogonal $U:m\times m$, orthogonal $V:n\times n$, and rectangular diagonal $\Sigma:m\times n$.

- The first $r$ columns of $V$ span the row space; remaining columns span the nullspace.
- The first $r$ columns of $U$ span the column space; remaining columns span the left nullspace.
- Singular values satisfy $\sigma_1\ge\sigma_2\ge\cdots\ge0$; rank $r$ counts strictly positive values in exact arithmetic.

The orthogonal factors can include reflections, not just rotations.
For an invertible square map, the spectral condition number is $\sigma_1/\sigma_n$.
Null directions prevent unrestricted inversion; $\sigma_1/\sigma_r$ describes conditioning only on the retained nonzero subspace.

## Truncated SVD and low-rank approximation

For rank at most $k$, a best approximation in Frobenius or spectral norm is

$$
A_k = U_k \Sigma_k V_k^\top
$$

Keep the leading $k$ singular values and their corresponding directions.
Orthogonal changes of basis preserve these norms, reducing the error calculation to discarded diagonal entries:

$$
\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2,\qquad
\|A-A_k\|_2=\sigma_{k+1}.
$$

Take omitted singular values as zero when none remain.
This is the Eckart–Young result; a low-rank model constraint does not automatically prescribe truncating an unrelated matrix.

## PCA as SVD

For $n>1$ observations in rows, fit a training mean $\mu$ and center $\widetilde X=X-\mathbf1\mu^\top$.

1. Decompose $\widetilde X=U\Sigma V^\top$.
2. Use leading columns $V_k$ as feature-space principal directions.
3. Compute scores $Z=\widetilde XV_k$ and reconstruct $\widehat X=ZV_k^\top+\mathbf1\mu^\top$.
4. Component sample variance is $\sigma_i^2/(n-1)$.

The covariance is $\widetilde X^\top\widetilde X/(n-1)$; its eigenvectors are these same principal directions.
Direct SVD avoids explicitly forming a Gram matrix, which squares the nonzero singular-value condition ratio.

For an exact numerical companion to the schematic, use rows $(-2,-1),(-2,1),(2,-1),(2,1)$.
They are centered, and $X^\top X=\operatorname{diag}(16,4)$, giving singular values $4,2$ and $V=I$.
Rank-one scores are $(-2,-2,2,2)^\top$; reconstruct each second coordinate as zero.
Four unit residuals give squared error $4$, while retained variance is $16/(16+4)=80\%$.

<!-- visual:pca-rank-one-projection -->
<figure class="learning-figure editorial-illustration" data-illustration-print-page aria-labelledby="pca-rank-one-projection-edition-title"><span class="visual-fragment-anchor" id="pca-projection-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="pca-svg-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="pca-svg-desc" aria-hidden="true"></span><p class="visual-title" id="pca-rank-one-projection-edition-title">Keep the coordinate along PC1; discard the perpendicular reconstruction error</p><div class="visual-argument" data-visual-argument="pca-rank-one-projection" data-argument-kind="geometry"><p class="visual-argument-premise">Keep one coordinate; discard the perpendicular residual</p><svg viewBox="0 0 360 425" role="img" aria-label="Keep one coordinate; discard the perpendicular residual. Projecting (3,2) onto the horizontal first principal direction retains (3,0). The two-unit residual is reconstruction error, not another retained coordinate."><title>Keep one coordinate; discard the perpendicular residual</title><desc>Projecting (3,2) onto the horizontal first principal direction retains (3,0). The two-unit residual is reconstruction error, not another retained coordinate.</desc><path d="M32 256.8L330 256.8" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M97.2 26L97.2 322" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="328" y="275.8" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="328" dy="0">x</tspan></text><text x="107.2" y="25" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="107.2" dy="0">y</tspan></text><path d="M97.2 256.8L262.8 146.4" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M258.2285237151706 154.15252246544583L262.8 146.4" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M253.88556530696178 147.63808485313263L262.8 146.4" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle cx="262.8" cy="146.4" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M97.2 256.8L262.8 256.8" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M254.6959760788259 260.7146898070011L262.8 256.8" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M254.6959760788259 252.88531019299893L262.8 256.8" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><circle cx="262.8" cy="256.8" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M262.8 256.8L262.8 146.4" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M20 349L45 349" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="57" y="353" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">Original centered point: (3, 2)</tspan></text><path d="M20 381L45 381" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="57" y="385" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">Rank-one reconstruction: (3, 0)</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 607" role="img" aria-labelledby="pca-rank-one-projection-edition-svg-title pca-rank-one-projection-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="pca-rank-one-projection-edition-svg-title">The retained coordinate stays; the perpendicular coordinate becomes reconstruction error</title><desc id="pca-rank-one-projection-edition-svg-desc">The original six centered toy samples are drawn at PC coordinates (130,10), (-130,-10), (85,-20), (-85,20), (40,10), and (-40,-10). Orthogonal PC axes are rotated thirty degrees. Each circle connects perpendicularly to its diamond projection. The selected sample x equals 85 v1 minus 20 v2; a right-angle marker identifies the discarded residual. A separate one-dimensional axis draws all six PC1 scores and highlights the same scalar z=85, distinct from the reconstructed vector 85 v1. Cross-covariance is zero and PC1 variance is larger. Squared discarded coordinates sum to twelve hundred.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Same six original toy samples; centered mean 0</tspan></text><text x="180" y="62" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Circle: sample; diamond: rank-1 reconstruction</tspan></text><path d="M58.53993711923248 288.125L301.4600628807675 147.875" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M148.125 162.79088050874202L211.875 273.209119491258" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><g data-pca-along="130" data-pca-residual="10"><path d="M279.9458071181805 170.11121593216774L275.6958071181805 162.75" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="279.9458071181805" cy="170.11121593216774" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M275.6958071181805 157.75L280.6958071181805 162.75L275.6958071181805 167.75L270.6958071181805 162.75Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g><g data-pca-along="-130" data-pca-residual="-10"><path d="M80.05419288181953 265.88878406783226L84.30419288181953 273.25" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="80.05419288181953" cy="265.88878406783226" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M84.30419288181953 268.25L89.30419288181953 273.25L84.30419288181953 278.25L79.30419288181953 273.25Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g><g data-pca-along="85" data-pca-residual="-20" data-pca-selected><path d="M234.0703354234257 167.15256813566455L242.5703354234257 181.875" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="234.0703354234257" cy="167.15256813566455" r="7" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:3"></circle><path d="M242.5703354234257 176.875L247.5703354234257 181.875L242.5703354234257 186.875L237.5703354234257 181.875Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g><path data-pca-right-angle d="M236.6813626776915 185.275L233.2813626776915 179.38602725426583L239.17033542342568 175.98602725426582" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5"></path><text x="222.0703354234257" y="152.15256813566455" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="222.0703354234257" dy="0">x</tspan></text><text x="274.5703354234257" y="201.875" style="font-size:12px;font-weight:400;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="274.5703354234257" dy="0">85 v1</tspan></text><g data-pca-along="-85" data-pca-residual="20"><path d="M125.92966457657431 268.8474318643355L117.42966457657431 254.125" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="125.92966457657431" cy="268.8474318643355" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M117.42966457657431 249.125L122.42966457657431 254.125L117.42966457657431 259.125L112.42966457657431 254.125Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g><g data-pca-along="40" data-pca-residual="10"><path d="M213.6948637286709 208.36121593216774L209.4448637286709 201" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="213.6948637286709" cy="208.36121593216774" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M209.4448637286709 196L214.4448637286709 201L209.4448637286709 206L204.4448637286709 201Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g><g data-pca-along="-40" data-pca-residual="-10"><path d="M146.3051362713291 227.63878406783226L150.5551362713291 235" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><circle cx="146.3051362713291" cy="227.63878406783226" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M150.5551362713291 230L155.5551362713291 235L150.5551362713291 240L145.5551362713291 235Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g><text x="273" y="120" style="font-size:12px;font-weight:400;fill:var(--viz-input-stroke);text-anchor:middle"><tspan x="273" dy="0">PC1: retain</tspan></text><text x="112" y="145" style="font-size:12px;font-weight:400;fill:var(--viz-state-stroke);text-anchor:middle"><tspan x="112" dy="0">PC2: discard</tspan></text><text x="180" y="243" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">0</tspan></text><text x="180" y="340" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">THE SAME PC1 COORDINATES, ON ONE AXIS</tspan></text><path d="M40 400L320 400" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle data-pca-score="130" cx="290.5" cy="400" r="3" style="fill:var(--viz-input-stroke)"></circle><circle data-pca-score="-130" cx="69.5" cy="400" r="3" style="fill:var(--viz-input-stroke)"></circle><circle data-pca-score="85" cx="252.25" cy="400" r="6" style="fill:var(--viz-focus-stroke)"></circle><text x="252.25" y="378" style="font-size:14px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="252.25" dy="0">z = 85</tspan></text><circle data-pca-score="-85" cx="107.75" cy="400" r="3" style="fill:var(--viz-input-stroke)"></circle><circle data-pca-score="40" cx="214" cy="400" r="3" style="fill:var(--viz-input-stroke)"></circle><circle data-pca-score="-40" cx="146" cy="400" r="3" style="fill:var(--viz-input-stroke)"></circle><path d="M52.5 395L52.5 405" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="52.5" y="430" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="52.5" dy="0">-150</tspan></text><path d="M180 395L180 405" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="430" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">0</tspan></text><path d="M307.5 395L307.5 405" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="307.5" y="430" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="307.5" dy="0">150</tspan></text><text x="180" y="470" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">z is one number; 85 v1 is a reconstructed</tspan><tspan x="180" dy="19">vector.</tspan></text><text x="180" y="524" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">x = 85 v1 - 20 v2; discard only the</tspan><tspan x="180" dy="19">perpendicular part.</tspan></text><text x="180" y="578" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">This error: 20^2 = 400. All six errors: 1200.</tspan></text></svg><div class="visual-derivation-caption"><strong>Read it this way:</strong> Follow the highlighted sample to scalar score z=85. Reconstruct the diamond at 85 v1. The perpendicular -20 v2 is discarded, giving squared error 400. These are illustrative coordinates, not measured data.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="pca-rank-one-projection"><strong>Check:</strong> Which object is the one-dimensional coordinate? How does it reconstruct the point on the PC1 line?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="pca-rank-one-projection">Projecting (3,2) onto the horizontal first principal direction retains (3,0). The two-unit residual is reconstruction error, not another retained coordinate.</span></figcaption></figure>

## Change the Requirement

### Retain Ninety Percent of Variance

**New task:** Choose the smallest rank retaining at least $90\%$ of total squared singular-value mass.

**Picture and change:** Accumulate $\sigma_i^2$ in descending order; change scores and reconstruction dimensions together.

**Check:** The fixture's first component retains only $80\%$; rank two retains $100\%$ and reconstructs exactly.

**Use when:** A declared reconstruction requirement determines storage or representation size.

### Standardize the Features

**New task:** Give both features equal sample variance before fitting PCA.

**Picture and change:** Divide centered columns by their training sample standard deviations, changing the feature-space metric.

**Check:** Fixture covariance becomes $I$; every unit direction is a possible first component, retaining $50\%$.

**Use when:** Units or domain priorities justify equalized feature weighting.
Carry the fitted mean, scales, and directions into inference; refitting on test data changes the transformation.

<!-- learning:core-end -->
<div data-learning-extension>

## Common pitfalls

- **Different objective.** Uncentered SVD includes the mean's contribution; its leading direction need not equal the mean direction.
- **Automatic scaling.** Standardization changes the objective and is undefined for zero-variance columns without a policy.
- **Whitening confusion.** PCA scores are uncorrelated, not generally unit variance.
  Dividing by positive component standard deviations whitens them; zero singular values need exclusion or regularization.
- **Direction ambiguity.** Singular-vector signs and tied-eigenvalue bases can vary without changing the reconstructed subspace.
- **Data geometry.** Centering sparse matrices can destroy sparsity; categorical data need a justified metric.

## Reconstruct

<details>
<summary>What if the label is the sign of the second feature?</summary>

The rank-one fixture discards that feature completely despite retaining eighty percent of variance.
Unsupervised reconstruction quality is not a guarantee of predictive sufficiency.

</details>

</div>

## Related

- [Matrices as linear maps](/concepts/matrices-as-linear-maps/). The geometry.
- [Eigenvalues and the spectral theorem](/concepts/eigenvalues-and-spectral-theorem/). For symmetric matrices.
