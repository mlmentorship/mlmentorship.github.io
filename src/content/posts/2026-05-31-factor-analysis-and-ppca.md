---
title: "Factor analysis and probabilistic PCA"
description: "Factor analysis uses latent factors with per-feature noise. Probabilistic PCA uses isotropic noise and recovers classical PCA in its zero-noise limit."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Factor analysis explains shared covariance through latent Gaussian factors and feature-specific measurement noise.
Probabilistic PCA ties those noise variances together.
Recognize the model when several measurements share variation but have different residual uncertainties.

Given two measurements, infer one hidden factor without treating either measurement as exact.
Use the teaching fixture $W=(2,1)^\top$, zero mean, noise variances $(1,4)$, and observation $x=(2,1)^\top$.

## The generative model

Let $z\in\mathbb R^k$, $x\in\mathbb R^d$, and $W\in\mathbb R^{d\times k}$.
The factor and measurement noise are independent:

$$
\mathbf{z} \sim \mathcal{N}(0, I), \qquad \mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(W\mathbf{z} + \boldsymbol{\mu},\ \Psi).
$$

$\mu$ is the observation mean; $\Psi$ is diagonal with strictly positive entries.
The loadings $W$ send each factor into the measured features.
Marginalizing the factors gives

$$
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu},\ WW^\top + \Psi).
$$

<!-- visual:factor-model-covariance-decomposition -->
<figure class="learning-figure" aria-labelledby="factor-covariance-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="factor-covariance-title">Share latent covariance, weight noisy measurements, and recompute when residual noise is tied</p><div class="visual-argument" data-visual-argument="factor-model-covariance-decomposition" data-argument-kind="matrix" data-argument-composition="shared-factor-area"><svg viewBox="0 0 360 2250" role="img" aria-label="Share latent covariance, weight noisy measurements, and recompute when residual noise is tied. One Gaussian factor creates shared covariance. Independent measurement noise adds only diagonal variance. Factor analysis allows different noise scales; PPCA ties them. For this fixture, the noisier measurement contributes less posterior precision. The prior shrinks the inferred factor toward zero. Tying the noise changes the posterior; reusing the old estimate is invalid. The coordinate remains uncertain and is not a causal explanation."><title>Share latent covariance, weight noisy measurements, and recompute when residual noise is tied</title><desc>One Gaussian factor creates shared covariance. Independent measurement noise adds only diagonal variance. Factor analysis allows different noise scales; PPCA ties them. For this fixture, the noisier measurement contributes less posterior precision. The prior shrinks the inferred factor toward zero. Tying the noise changes the posterior; reusing the old estimate is invalid. The coordinate remains uncertain and is not a causal explanation.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Infer one uncertain factor from two</tspan><tspan x="20" dy="18">noisy measurements</tspan></text><text x="20" y="105" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">One standard Gaussian factor; zero</tspan><tspan x="20" dy="18">observation mean</tspan></text><circle cx="180" cy="205" r="31" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><text x="180" y="200" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">latent</tspan><tspan x="180" dy="18">factor</tspan></text><path d="M158 245L80 310M83.58503263812108 301.8056396842947L80 310L88.70650783543692 307.95140992107366" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="80" cy="350" r="27" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="80" y="356" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="80" dy="0">2</tspan></text><text x="80" y="420" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="80" dy="0">Loading 2; noise</tspan><tspan x="80" dy="18">variance 1</tspan></text><path d="M202 245L280 310M271.2934921645631 307.95140992107366L280 310L276.4149673618789 301.8056396842947" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="280" cy="350" r="27" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="280" y="356" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="280" dy="0">1</tspan></text><text x="280" y="420" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="280" dy="0">Loading 1; noise</tspan><tspan x="280" dy="18">variance 4</tspan></text><text x="20" y="540" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Shared covariance</tspan></text><rect x="105" y="580" width="70" height="55" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="140" y="615" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="140" dy="0">4</tspan></text><rect x="183" y="580" width="70" height="55" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="218" y="615" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">2</tspan></text><rect x="105" y="643" width="70" height="55" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="140" y="678" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="140" dy="0">2</tspan></text><rect x="183" y="643" width="70" height="55" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="218" y="678" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">1</tspan></text><text x="20" y="780" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Independent noise adds only to</tspan><tspan x="20" dy="18">diagonal entries: one and four.</tspan></text><text x="20" y="890" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Total covariance</tspan></text><rect x="105" y="930" width="70" height="55" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="140" y="965" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="140" dy="0">5</tspan></text><rect x="183" y="930" width="70" height="55" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="218" y="965" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">2</tspan></text><rect x="105" y="993" width="70" height="55" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="140" y="1028" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="140" dy="0">2</tspan></text><rect x="183" y="993" width="70" height="55" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="218" y="1028" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">5</tspan></text><text x="20" y="1140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Posterior precision: prior plus both</tspan><tspan x="20" dy="18">measurements</tspan></text><text x="20" y="1240" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Prior: 1</tspan></text><rect x="20" y="1270" width="80" height="30" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="20" y="1380" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">First measurement: 4</tspan></text><rect x="20" y="1410" width="320" height="30" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1520" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Noisier second: 0.25</tspan></text><rect x="20" y="1550" width="20" height="30" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="20" y="1690" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Posterior mean 0.810; variance 0.190</tspan></text><text x="20" y="1790" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">The prior shrinks the unregularized</tspan><tspan x="20" dy="18">coordinate one toward zero.</tspan></text><text x="20" y="1895" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">PPCA ties residual noise. Set both</tspan><tspan x="20" dy="18">variances to one, keeping loadings</tspan><tspan x="20" dy="18">and observations fixed.</tspan></text><text x="20" y="2020" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Recomputed mean 0.833; variance 0.167</tspan></text><text x="20" y="2120" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Do not reuse the old posterior.</tspan><tspan x="20" dy="18">Classical PCA instead minimizes</tspan><tspan x="20" dy="18">reconstruction error; a latent name</tspan><tspan x="20" dy="18">is not causal evidence.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 500" role="img" aria-labelledby="factor-covariance-svg-title factor-covariance-svg-desc">
			<title id="factor-covariance-svg-title">Factor-model graph and covariance decomposition</title>
			<desc id="factor-covariance-svg-desc">A shared latent factor z reaches three features through loading arrows. Independent noises reach only their own features. The covariance is W W transpose plus diagonal Psi. Factor analysis permits unequal noise variances; probabilistic PCA ties them. Fitted PPCA has PCA's principal subspace at finite noise. Deterministic projection is a separate zero-noise limit.</desc>
			<defs><marker id="factor-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker></defs>
			<text class="viz-axis-label" x="12" y="18">GENERATIVE VIEW · xᵢ = wᵢᵀz + μᵢ + εᵢ</text>
			<rect class="viz-node viz-node--focus" x="138" y="32" width="84" height="44" rx="5"></rect>
			<text class="viz-node-label" x="180" y="53">latent z</text><text class="viz-node-value" x="180" y="68">shared cause</text>
			<path d="M157 76L62 121M180 76V121M203 76L298 121" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#factor-arrow)"></path>
			<text class="viz-edge-label" x="104" y="94">loading w₁</text><text class="viz-edge-label" x="180" y="100">w₂</text><text class="viz-edge-label" x="256" y="94">w₃</text>
			<rect class="viz-node viz-node--output" x="20" y="124" width="84" height="44" rx="5"></rect><rect class="viz-node viz-node--output" x="138" y="124" width="84" height="44" rx="5"></rect><rect class="viz-node viz-node--output" x="256" y="124" width="84" height="44" rx="5"></rect>
			<text class="viz-node-label" x="62" y="151">feature x₁</text><text class="viz-node-label" x="180" y="151">feature x₂</text><text class="viz-node-label" x="298" y="151">feature x₃</text>
			<rect class="viz-node viz-node--input" x="20" y="199" width="84" height="38" rx="5"></rect><rect class="viz-node viz-node--input" x="138" y="199" width="84" height="38" rx="5"></rect><rect class="viz-node viz-node--input" x="256" y="199" width="84" height="38" rx="5"></rect>
			<text class="viz-node-value" x="62" y="222">unique noise ε₁</text><text class="viz-node-value" x="180" y="222">unique noise ε₂</text><text class="viz-node-value" x="298" y="222">unique noise ε₃</text>
			<path d="M62 199V172M180 199V172M298 199V172" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:5 4;marker-end:url(#factor-arrow)"></path>
			<text class="viz-axis-label" x="12" y="269">MARGINAL VIEW · Cov(x) = WWᵀ + Ψ</text>
			<text class="viz-callout" x="15" y="331">=</text>
			<rect class="viz-node--focus" x="42" y="284" width="34" height="34"></rect><rect class="viz-node--focus" x="76" y="284" width="34" height="34"></rect><rect class="viz-node--focus" x="110" y="284" width="34" height="34"></rect>
			<rect class="viz-node--focus" x="42" y="318" width="34" height="34"></rect><rect class="viz-node--focus" x="76" y="318" width="34" height="34"></rect><rect class="viz-node--focus" x="110" y="318" width="34" height="34"></rect>
			<rect class="viz-node--focus" x="42" y="352" width="34" height="34"></rect><rect class="viz-node--focus" x="76" y="352" width="34" height="34"></rect><rect class="viz-node--focus" x="110" y="352" width="34" height="34"></rect>
			<text class="viz-edge-label" x="59" y="305">w₁·w₁</text><text class="viz-edge-label" x="93" y="305">w₁·w₂</text><text class="viz-edge-label" x="127" y="305">w₁·w₃</text>
			<text class="viz-edge-label" x="59" y="339">w₂·w₁</text><text class="viz-edge-label" x="93" y="339">w₂·w₂</text><text class="viz-edge-label" x="127" y="339">w₂·w₃</text>
			<text class="viz-edge-label" x="59" y="373">w₃·w₁</text><text class="viz-edge-label" x="93" y="373">w₃·w₂</text><text class="viz-edge-label" x="127" y="373">w₃·w₃</text>
			<text class="viz-callout" x="157" y="339">+</text>
			<rect class="viz-node--input" x="178" y="284" width="34" height="34"></rect><rect class="viz-node" x="212" y="284" width="34" height="34"></rect><rect class="viz-node" x="246" y="284" width="34" height="34"></rect>
			<rect class="viz-node" x="178" y="318" width="34" height="34"></rect><rect class="viz-node--input" x="212" y="318" width="34" height="34"></rect><rect class="viz-node" x="246" y="318" width="34" height="34"></rect>
			<rect class="viz-node" x="178" y="352" width="34" height="34"></rect><rect class="viz-node" x="212" y="352" width="34" height="34"></rect><rect class="viz-node--input" x="246" y="352" width="34" height="34"></rect>
			<text class="viz-edge-label" x="195" y="305">ψ₁</text><text class="viz-edge-label" x="229" y="305">0</text><text class="viz-edge-label" x="263" y="305">0</text>
			<text class="viz-edge-label" x="195" y="339">0</text><text class="viz-edge-label" x="229" y="339">ψ₂</text><text class="viz-edge-label" x="263" y="339">0</text>
			<text class="viz-edge-label" x="195" y="373">0</text><text class="viz-edge-label" x="229" y="373">0</text><text class="viz-edge-label" x="263" y="373">ψ₃</text>
			<text class="viz-axis-label" x="93" y="403" text-anchor="middle">WWᵀ · shared covariance</text><text class="viz-axis-label" x="229" y="403" text-anchor="middle">Ψ · unique variance only</text>
			<rect class="viz-node viz-node--focus" x="8" y="429" width="104" height="58" rx="5"></rect><rect class="viz-node viz-node--input" x="128" y="429" width="104" height="58" rx="5"></rect><rect class="viz-node" x="248" y="429" width="104" height="58" rx="5"></rect>
			<text class="viz-node-label" x="60" y="450">FA</text><text class="viz-node-value" x="60" y="468">ψ₁, ψ₂, ψ₃</text><text class="viz-node-value" x="60" y="481">may differ</text>
			<text class="viz-node-label" x="180" y="450">PPCA</text><text class="viz-node-value" x="180" y="468">ψᵢ = σ²</text><text class="viz-node-value" x="180" y="481">same for every i</text>
			<text class="viz-node-label" x="300" y="450">PCA limit</text><text class="viz-node-value" x="300" y="468">σ² → 0</text><text class="viz-node-value" x="300" y="481">no noise</text>
		</svg>
	</div>
	<table><caption>Two-feature teaching fixture</caption><thead><tr><th>Shared WWᵀ</th><th>Noise Ψ</th><th>Total C</th></tr></thead><tbody><tr><td>[[4, 2], [2, 1]]</td><td>[[1, 0], [0, 4]]</td><td>[[5, 2], [2, 5]]</td></tr></tbody></table>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> shared factor paths create both variance and covariance; each noise path adds only its feature's variance. The table applies that decomposition to two features. PPCA ties noise variances, but its fitted principal subspace already matches PCA at finite noise. Deterministic projection is a separate zero-noise limit. The original drawing retains its Tipping-Bishop (1999) and Ghahramani-Hinton (1996) provenance.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="factor-model-covariance-decomposition"><strong>Check:</strong> Why can independent unique noise alter variances but not explain cross-feature covariance?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="factor-model-covariance-decomposition">One Gaussian factor creates shared covariance. Independent measurement noise adds only diagonal variance. Factor analysis allows different noise scales; PPCA ties them. For this fixture, the noisier measurement contributes less posterior precision. The prior shrinks the inferred factor toward zero. Tying the noise changes the posterior; reusing the old estimate is invalid. The coordinate remains uncertain and is not a causal explanation. <strong>Does tying the noise preserve the old posterior mean?</strong></span></figcaption>
</figure>

### Work the posterior

The shared contribution is $WW^\top=\begin{bmatrix}4&2\\2&1\end{bmatrix}$.
Adding $\Psi$ gives $C=\begin{bmatrix}5&2\\2&5\end{bmatrix}$, whose eigenvalues are $7$ and $3$.
The off-diagonal value $2$ comes entirely from the shared factor.

Completing the square in the Gaussian log density gives posterior covariance $S$ and mean $m$:

$$
S=(I+W^\top\Psi^{-1}W)^{-1},\qquad
m=SW^\top\Psi^{-1}(x-\mu).
$$

Here the prior contributes precision $1$; the measurements contribute $4$ and $1/4$.
Therefore

$$
S=(1+4+1/4)^{-1}=4/21,\qquad
m=(4/21)(4+1/4)=17/21.
$$

The noisier second feature supplies less precision.
The prior shrinks the mean below the unregularized projection coefficient $1$.
Neither the inferred coordinate nor its name establishes a causal explanation.

## FA vs PPCA vs PCA: it's all about $\Psi$

| Model | Noise covariance $\Psi$ | Consequence |
| --- | --- | --- |
| Factor analysis | Positive diagonal | Separate residual variances |
| Probabilistic PCA | $\sigma^2I$ | Shared residual variance; spectral likelihood fit |
| Classical PCA | No probabilistic noise requirement | Minimum squared reconstruction error |

### Tie the measurement noise

**New task:** require $\Psi=I$ while keeping $W$ and $x$ fixed.

**Picture and change:** replace the unequal noise entries, then recompute both covariance and posterior precision.

**Check:** $C=\begin{bmatrix}5&2\\2&2\end{bmatrix}$, $S=1/6$, and $m=5/6$.
Reusing $17/21$ would retain the wrong measurement weights.

**Use when:** one shared residual scale is an intended modeling restriction.

### Predict an unobserved feature

**New task:** observe only $x_1=2$ and predict $x_2$.

**Picture and change:** condition using only the observed loading row and noise entry.
Keep the missing feature's independent noise when predicting it.

**Check:** $z\mid x_1$ has mean $4/5$ and variance $1/5$.
Consequently, $x_2\mid x_1$ has mean $4/5$ and variance $1/5+4=21/5$.
Using only $1/5$ would omit future measurement noise.

**Use when:** incomplete observations have a justified observation model; missingness need not be harmless.

<!-- learning:core-end -->

<div data-learning-extension>

## Fitting it

For $0<k<d$, order the sample covariance eigenvalues $\lambda_1\geq\cdots\geq\lambda_d$.
At a regular PPCA likelihood solution,

$$
\widehat\sigma^2=\frac{\sum_{j=k+1}^d\lambda_j}{d-k},\qquad
\widehat W=U_k(\Lambda_k-\widehat\sigma^2I)^{1/2}R,
$$

where $R$ is orthogonal.
For our covariance, $\widehat\sigma^2=3$ and $\widehat W=\sqrt2(1,1)^\top$.
The principal subspace already matches PCA; positive noise does not prevent that agreement.
Sending noise to zero separately can make the observation density singular.
FA generally uses iterative likelihood fitting, including [EM](/concepts/expectation-maximization/).

## Why the probabilistic version is worth it

Likelihood supports held-out comparison, not automatic rank selection without an overfitting policy.
Marginals support missing-feature prediction; the generative model supports sampling and mixtures of local factor models.
[VAEs](/concepts/variational-autoencoders/) extend the decoder and inference family, not the exact Gaussian conditioning identity.

## What an interviewer expects you to say

Rescale $x_1$ by ten. Which parameters must change?

<details><summary>Answer</summary>

With $D=\operatorname{diag}(10,1)$, use $DW$, $D\mu$, $D\Psi D^\top$, and $DCD^\top$.
An unrestricted diagonal-noise family remains compatible; fixed penalties or constraints may not.
Isotropic noise generally stops being isotropic.

</details>

## Common confusions

For orthogonal $R$, $W$ and $WR$ have identical observed covariance.
Factor axes are not uniquely identified causes; even one factor permits a sign reversal.

</div>

---

*Related: [SVD and PCA](/concepts/svd-and-pca/), [Expectation-maximization](/concepts/expectation-maximization/), [Gaussian mixture models](/concepts/gaussian-mixture-models/), [Variational autoencoders](/concepts/variational-autoencoders/).*
