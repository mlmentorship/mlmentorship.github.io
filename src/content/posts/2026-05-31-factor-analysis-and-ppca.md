---
title: "Factor analysis and probabilistic PCA"
description: "Factor analysis uses latent factors with per-feature noise. Probabilistic PCA uses isotropic noise and recovers classical PCA in its zero-noise limit."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Factor analysis (FA) is a **latent linear-Gaussian model**: each observation is a linear map of a few low-dimensional latent factors plus Gaussian noise. **Probabilistic PCA (PPCA)** is the special case with **isotropic** noise, and classical PCA falls out as its zero-noise / maximum-likelihood limit.

This is the model that turns PCA from "an eigen-decomposition trick" into "a probabilistic generative model," which is the framing senior interviewers want. It connects dimensionality reduction to the EM algorithm, to VAEs (a nonlinear PPCA), and to the generative-vs-discriminative discussion. It's also a clean example of how a **prior + likelihood** recovers a classical algorithm as a limiting case.

## The generative model

Latent factor $\mathbf{z} \in \mathbb{R}^k$ with $k \ll d$, observation $\mathbf{x} \in \mathbb{R}^d$:

$$
\mathbf{z} \sim \mathcal{N}(0, I), \qquad \mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(W\mathbf{z} + \boldsymbol{\mu},\ \Psi).
$$

$W \in \mathbb{R}^{d\times k}$ is the **factor loading matrix** (the directions), and $\Psi$ is the noise covariance. Marginalizing $\mathbf{z}$ gives a Gaussian with **low-rank-plus-structured** covariance:

$$
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu},\ WW^\top + \Psi).
$$

<!-- visual:factor-model-covariance-decomposition -->
<figure class="learning-figure" aria-labelledby="factor-covariance-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="factor-covariance-title">Separate covariance shared through the factors from noise unique to each feature</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 500" role="img" aria-labelledby="factor-covariance-svg-title factor-covariance-svg-desc">
			<title id="factor-covariance-svg-title">Factor-model graph and covariance decomposition</title>
			<desc id="factor-covariance-svg-desc">A shared latent factor z has solid arrows to observed features x1, x2, and x3. Separate noises epsilon1, epsilon2, and epsilon3 have dashed arrows to one feature each. Below, the covariance of x is decomposed into W W transpose, a full three-by-three matrix that creates variance and cross-feature covariance, plus Psi, a diagonal matrix that adds only feature-specific variance. Factor analysis permits different diagonal noise values, probabilistic PCA uses one shared sigma squared value, and classical PCA is the zero-noise limit.</desc>
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
	<figcaption><strong>Read it this way:</strong> follow the solid arrows first: one latent vector reaches every feature, so WWᵀ can fill both diagonal and off-diagonal covariance cells. Each dashed noise path reaches only one feature, so Ψ adds variance only on the diagonal. FA lets those diagonal additions differ; PPCA ties them to one σ²; PCA is the zero-noise limit. The construction is original and checked against Tipping and Bishop (1999) and Ghahramani and Hinton (1996).</figcaption>
</figure>

The whole model is the claim: *the correlations between observed variables are explained by a few shared latent factors; whatever is left is independent per-feature noise.*

## FA vs PPCA vs PCA: it's all about $\Psi$

| Model | Noise covariance $\Psi$ | Consequence |
| --- | --- | --- |
| **Factor analysis** | **diagonal** $\text{diag}(\psi_1,\dots,\psi_d)$ | per-feature noise; **scale-invariant**; models unique variances |
| **Probabilistic PCA** | **isotropic** $\sigma^2 I$ | one shared noise level; MLE has closed form via eigendecomposition |
| **Classical PCA** | $\sigma^2 \to 0$ limit | deterministic projection onto top-$k$ eigenvectors |

For interviews, distinguish the noise models: **FA has diagonal noise covariance, while PPCA uses the same isotropic noise for every feature.** FA is invariant to rescaling individual features. PCA and PPCA are sensitive to feature scaling, which is why inputs are usually standardized first.

## Fitting it

- **PPCA** has a **closed-form MLE**: $W$ is recovered from the top-$k$ eigenvectors of the sample covariance scaled by $(\lambda_i - \sigma^2)^{1/2}$, with $\sigma^2$ = average of the discarded eigenvalues. So PPCA ≈ PCA plus a noise estimate.
- **FA** has no closed form (the diagonal $\Psi$ couples things); it's fit with **EM**: the E-step infers the posterior over factors $p(\mathbf{z}\mid\mathbf{x})$, the M-step updates $W$ and $\Psi$. This is a textbook EM application.

## Why the probabilistic version is worth it

Recasting PCA as a model buys you things plain PCA can't do:

- A proper **likelihood** → principled model comparison and a way to choose $k$.
- Natural handling of **missing data** (marginalize unobserved dimensions in EM).
- A generative model you can **sample** from.
- **Mixtures of PPCA/FA** for non-linear, multi-modal structure.
- The conceptual bridge to the **VAE**, which is "PPCA with a neural-network decoder and amortized inference."

## What an interviewer expects you to say

1. Write the **latent linear-Gaussian generative model** and the marginal covariance $WW^\top + \Psi$.
2. State the difference: **FA = diagonal noise, PPCA = isotropic noise, PCA = zero-noise limit of PPCA**.
3. Explain the practical consequence: **FA is scale-invariant; PCA/PPCA require feature standardization**.
4. Know that **PPCA has a closed-form (eigendecomposition) MLE** while **FA needs EM**.
5. Bonus: connect to **VAEs** (nonlinear PPCA) and note the probabilistic framing enables missing data, model selection, and sampling.

## Common confusions

- **"FA and PCA are the same."** FA models per-feature (diagonal) noise and explains *covariance*; PCA maximizes *retained variance* and assumes isotropic/zero noise. They give different loadings unless noise is uniform.
- **"PPCA is fancier PCA with no payoff."** The payoff is the likelihood: model selection, missing data, sampling, mixtures.
- **"The factors are unique."** $W$ is only identifiable up to rotation (you can rotate $\mathbf{z}$ and absorb it into $W$), hence "factor rotation" (varimax) for interpretability.
- **"FA needs scaling like PCA."** FA is invariant to per-feature rescaling because its diagonal noise absorbs scale; PCA is not.

---

*Related: [SVD and PCA](/concepts/svd-and-pca/), [Expectation-maximization](/concepts/expectation-maximization/), [Gaussian mixture models](/concepts/gaussian-mixture-models/), [Variational autoencoders](/concepts/variational-autoencoders/).*
