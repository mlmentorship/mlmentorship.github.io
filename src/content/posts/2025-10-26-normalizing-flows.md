---
title: "Normalizing flows"
description: "Generative models built from invertible transformations. Compute exact likelihoods and sample efficiently. At the cost of architectural restrictions."
date: "2025-10-26"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **normalizing flow** transforms a simple base distribution (typically standard Gaussian) into a target distribution through a sequence of **invertible**, **differentiable** mappings $f_K \circ \dots \circ f_1$. The change-of-variables formula gives exact log-likelihood:

$$
\log p_X(x) = \log p_Z(f^{-1}(x)) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|.
$$

Flows are the only family of deep generative models that simultaneously offer:

- **Exact likelihoods** (unlike VAEs and diffusion, which give bounds).
- **Efficient sampling** (single forward pass, unlike diffusion's iterative denoise).
- **Tractable posterior** (the inverse function gives the exact $z$ for any $x$).

The architectural restriction (each layer must be invertible with a tractable Jacobian) limits expressiveness. Flows have been displaced by diffusion for high-fidelity image generation but remain useful for likelihood-critical applications: density estimation, anomaly detection, simulation-based inference, molecular generation.

## The change-of-variables formula

For an invertible $f$ with $z = f^{-1}(x)$:

$$
p_X(x) = p_Z(z) \cdot \left| \det J_{f^{-1}}(x) \right| = p_Z(z) \cdot \left| \det J_f(z) \right|^{-1}.
$$

For a composition of $K$ flows, the log-determinant decomposes additively:

$$
\log p_X(x) = \log p_Z(z) - \sum_{k=1}^{K} \log \left| \det J_{f_k}(z_{k-1}) \right|.
$$

<!-- visual:flow-local-volume-density -->
<figure class="learning-figure plot-panel" aria-labelledby="flow-volume-visual-title">
	<p class="visual-kicker">Change of variables</p>
	<p class="visual-title" id="flow-volume-visual-title">The same local probability mass spreads over more area, so its density falls.</p>
	<svg viewBox="0 0 360 630" role="img" aria-labelledby="flow-volume-svg-title flow-volume-svg-desc">
		<title id="flow-volume-svg-title">Local volume and density through two invertible flow layers</title>
		<desc id="flow-volume-svg-desc">A small patch around z zero has relative area A, probability mass m, and density p zero. The first invertible layer stretches the patch horizontally by two, so its Jacobian determinant is two, its area is 2A, and its density is p zero divided by two. The second layer stretches vertically by three halves, so the total area is 3A and the final density is p zero divided by three. Solid downward arrows mark forward sampling. Dashed upward arrows mark inverse density evaluation. The log-density ledger subtracts log two and log three halves, which equals subtracting log three.</desc>
		<defs>
			<marker id="flow-volume-forward-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-focus-stroke)"></path></marker>
			<marker id="flow-volume-inverse-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-edge)"></path></marker>
		</defs>
		<text class="viz-axis-label" x="18" y="24">1 · BASE SPACE z₀</text>
		<rect class="viz-node viz-node--input" x="150" y="42" width="60" height="60" rx="3"></rect>
		<g style="fill:var(--viz-input-stroke)">
			<circle cx="165" cy="58" r="3"></circle><circle cx="180" cy="58" r="3"></circle><circle cx="195" cy="58" r="3"></circle>
			<circle cx="165" cy="86" r="3"></circle><circle cx="180" cy="86" r="3"></circle><circle cx="195" cy="86" r="3"></circle>
		</g>
		<text class="viz-callout" x="180" y="119" text-anchor="middle">area A · mass m · density p₀ = m/A</text>
		<path d="M274 125V177" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;marker-end:url(#flow-volume-forward-arrow)"></path>
		<text class="viz-callout" x="268" y="145" text-anchor="end">sample forward</text>
		<text class="viz-label" x="268" y="163" text-anchor="end">f₁ · |det J₁| = 2</text>
		<path d="M86 177V125" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:5 4;marker-end:url(#flow-volume-inverse-arrow)"></path>
		<text class="viz-callout" x="92" y="145">evaluate density</text>
		<text class="viz-label" x="92" y="163">with f₁⁻¹</text>
		<text class="viz-axis-label" x="18" y="205">2 · AFTER HORIZONTAL STRETCH z₁</text>
		<rect class="viz-node viz-node--focus" x="120" y="222" width="120" height="60" rx="3"></rect>
		<g style="fill:var(--viz-focus-stroke)">
			<circle cx="150" cy="238" r="3"></circle><circle cx="180" cy="238" r="3"></circle><circle cx="210" cy="238" r="3"></circle>
			<circle cx="150" cy="266" r="3"></circle><circle cx="180" cy="266" r="3"></circle><circle cx="210" cy="266" r="3"></circle>
		</g>
		<text class="viz-callout" x="180" y="299" text-anchor="middle">area 2A · same mass m · density p₀/2</text>
		<path d="M274 305V357" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;marker-end:url(#flow-volume-forward-arrow)"></path>
		<text class="viz-callout" x="268" y="325" text-anchor="end">sample forward</text>
		<text class="viz-label" x="268" y="343" text-anchor="end">f₂ · |det J₂| = 3/2</text>
		<path d="M86 357V305" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:5 4;marker-end:url(#flow-volume-inverse-arrow)"></path>
		<text class="viz-callout" x="92" y="325">evaluate density</text>
		<text class="viz-label" x="92" y="343">with f₂⁻¹</text>
		<text class="viz-axis-label" x="18" y="385">3 · DATA SPACE x</text>
		<rect class="viz-node viz-node--output" x="120" y="402" width="120" height="90" rx="3"></rect>
		<g style="fill:var(--viz-output-stroke)">
			<circle cx="150" cy="426" r="3"></circle><circle cx="180" cy="426" r="3"></circle><circle cx="210" cy="426" r="3"></circle>
			<circle cx="150" cy="468" r="3"></circle><circle cx="180" cy="468" r="3"></circle><circle cx="210" cy="468" r="3"></circle>
		</g>
		<text class="viz-callout" x="180" y="510" text-anchor="middle">area 3A · same mass m · density pₓ = p₀/3</text>
		<rect class="viz-plot-bg" x="18" y="532" width="324" height="78" rx="6"></rect>
		<text class="viz-axis-label" x="32" y="553">LOG-DENSITY LEDGER</text>
		<text class="viz-callout" x="180" y="578" text-anchor="middle">log pₓ = log p₀ − log 2 − log(3/2)</text>
		<text class="viz-node-value" x="180" y="598" text-anchor="middle">= log p₀ − log 3</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the solid arrows down to sample: each bijection stretches the same six-point probability-mass patch. Read the dashed arrows up to evaluate density: divide by each layer's local volume multiplier. Here the areas multiply, <code>2 × 3/2 = 3</code>, while the log corrections add, <code>log 2 + log(3/2) = log 3</code>. This is an original local construction checked against <a href="https://arxiv.org/abs/1605.08803">Real NVP</a> and the <a href="https://jmlr.org/papers/v22/19-1028.html">JMLR normalizing-flows review</a>.</figcaption>
</figure>

The engineering challenge: design each $f_k$ to be (a) invertible, (b) expressive, and (c) have a **cheap-to-compute log-determinant**.

## Common flow families

| Family | Idea | Tradeoff |
|--------|------|----------|
| **Affine coupling** (NICE, RealNVP, Glow) | Split $x$ in half; one half passes through, the other is affinely transformed by a function of the first | Triangular Jacobian → $\det$ is product of diagonal; needs many layers for expressiveness |
| **Autoregressive** (MAF, IAF) | Each output dimension is an affine function of preceding ones; Jacobian is triangular | MAF: fast density, slow sample. IAF: fast sample, slow density. |
| **Continuous-time / Neural ODE** (FFJORD) | Define $f$ via $dx/dt = g_\theta(x, t)$ and integrate; Jacobian via Hutchinson trace estimator | Very expressive; expensive integration |
| **Invertible 1×1 convolutions** (Glow) | Permutation generalization for image flows | Used inside Glow for permutation between coupling layers |

## RealNVP / coupling layers (the workhorse)

Split $x = (x_a, x_b)$. Then:

$$
y_a = x_a, \qquad y_b = x_b \odot \exp(s(x_a)) + t(x_a)
$$

with neural nets $s, t$. The Jacobian is lower triangular with $\exp(s(x_a))$ on the diagonal of the $y_b$ block. Determinant: $\prod_i \exp(s(x_a)_i)$.

Stack many coupling layers, alternating which half passes through, with shuffles or 1×1 convs between them.

## When to use flows in 2026

| Setting | Flows vs. alternatives |
|---------|----------------------|
| High-fidelity image generation | Use diffusion; flows are non-competitive |
| Density estimation, OOD detection | Flows give exact likelihood |
| Simulation-based inference (likelihood-free) | Flows excellent (NPE, NRE) |
| Molecular conformation / coordinates | Flows used (E-NF, equivariant flows) |
| Probabilistic forecasting | Flows + RNN backbones (Real-NVP-style) |
| Variational inference posterior approx | Flows as flexible $q$ |

## Common pitfalls

- **Computing log-determinants without exploiting structure.** General log-det is $O(d^3)$. Always use a flow with a structured Jacobian (triangular, low-rank).
- **Confusing log $|\det|$ with log-likelihood directly.** The full formula has both the base density term and the determinant term.
- **Treating flows as fast-to-train.** They are usually slower per epoch than VAE / diffusion at matched parameter count due to expensive Jacobian computations.
- **Using flows on discrete data.** Flows assume continuous, differentiable spaces. For discrete: dequantize (add uniform noise), or use discrete normalizing flows (more complex).

## Related

- [Variational autoencoders](/concepts/variational-autoencoders/). Alternative latent-variable generative model.
- [Autoregressive vs. diffusion](/concepts/autoregressive-vs-diffusion/). Broader paradigm comparison.
