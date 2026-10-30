---
title: "Normalizing flows"
description: "Transport a base density through invertible maps and track local volume. Likelihood and sampling costs depend on the architecture."
date: "2025-10-26"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A normalizing flow transports a base density through invertible transformations, accounting for their local volume changes.
For the differentiable bijections here, density evaluation uses the inverse coordinate and the inverse Jacobian's absolute determinant:

$$
\log p_X(x) = \log p_Z(f^{-1}(x)) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|.
$$

An invertible coordinate is not a nontrivial VAE-style posterior distribution.
Tractable discrete-layer likelihood and fast sampling depend on the selected architecture; neither property uniquely belongs to flows.

## The change-of-variables formula

Require a differentiable inverse and nonzero Jacobian determinant on the relevant support.
For $z=f^{-1}(x)$:

$$
p_X(x) = p_Z(z) \cdot \left| \det J_{f^{-1}}(x) \right| = p_Z(z) \cdot \left| \det J_f(z) \right|^{-1}.
$$

For a composition of $K$ flows, the log-determinant decomposes additively:

$$
\log p_X(x) = \log p_Z(z) - \sum_{k=1}^{K} \log \left| \det J_{f_k}(z_{k-1}) \right|.
$$

Evaluate each layer's determinant at its actual intermediate state.
Evaluate standard Gaussian density at $z=(1,2)$.
Set $f_1(z)=(2z_1,z_2)$ and $f_2(u)=(u_1,1.5u_2)$.
The states are $(1,2)\to(2,2)\to(2,3)$, with determinant factors two and $1.5$.
The local area triples while probability mass is conserved, so $p_X(2,3)=p_Z(1,2)/3$.
Gaussian log density gives:

$$
\log p_X(2,3)=-\log(2\pi)-2.5-\log3\approx-5.43649.
$$

<!-- visual:flow-local-volume-density -->
<figure class="learning-figure plot-panel" aria-labelledby="flow-volume-visual-title">
	<p class="visual-kicker">Change of variables</p>
	<p class="visual-title" id="flow-volume-visual-title">Expanding volume lowers density while preserving probability mass</p><div class="visual-argument" data-visual-argument="flow-local-volume-density" data-argument-kind="areas"><p class="visual-argument-premise">Expanding volume dilutes density, not probability mass</p><svg viewBox="0 0 360 517" role="img" aria-label="Expanding volume dilutes density, not probability mass. The same four dots carry mass 1 before and after expansion. Doubling the drawn area changes density from 1 to 0.5, not the total probability mass."><title>Expanding volume dilutes density, not probability mass</title><desc>The same four dots carry mass 1 before and after expansion. Doubling the drawn area changes density from 1 to 0.5, not the total probability mass.</desc><text x="20" y="25" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Before expansion: four dots</tspan></text><g data-proof-region="0" data-proof-mass="1"><rect data-proof-area="1" data-proof-scale="120" x="20" y="57" width="120" height="120" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><circle cx="50" cy="87" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="110" cy="87" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="50" cy="147" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="110" cy="147" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><text x="20" y="203" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Volume 1; density 1</tspan></text><text x="20" y="257" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">After expansion: the same four dots</tspan></text><g data-proof-region="1" data-proof-mass="1"><rect data-proof-area="2" data-proof-scale="120" x="20" y="289" width="240" height="120" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="80" cy="319" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="200" cy="319" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="80" cy="379" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="200" cy="379" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="20" y="435" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Volume 2; density 0.5</tspan></text><text x="20" y="489" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Each dot = 0.25 probability mass</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> forward transforms stretch a local patch without changing its probability mass. Inverse density evaluation divides by the total volume multiplier. Here area multiplies by three and log density loses log three. Dots illustrate mass; they are not point-probability density values. Original construction follows <a href="https://arxiv.org/abs/1605.08803">Real NVP</a> and the <a href="https://jmlr.org/papers/v22/19-1028.html">flows review</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="flow-local-volume-density"><strong>Check:</strong> Why does tripling occupied volume divide density by three while leaving total probability mass unchanged?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="flow-local-volume-density">The same four dots carry mass 1 before and after expansion. Doubling the drawn area changes density from 1 to 0.5, not the total probability mass.</span></figcaption>
</figure>

Inverting $(2,3)$ returns $(1,2)$; multiplying its transformed density by tripled local area recovers the same mass.

## RealNVP / coupling layers (the workhorse)

Split $x = (x_a, x_b)$. Then:

$$
y_a = x_a, \qquad y_b = x_b \odot \exp(s(x_a)) + t(x_a)
$$

The Jacobian is block triangular, with log absolute determinant $\sum_i s(x_a)_i$.
Inverse reconstruction uses $x_a=y_a$ and $x_b=(y_b-t(y_a))\odot\exp(-s(y_a))$.
The internal s/t networks need not themselves be invertible.

Stack many coupling layers, alternating which half passes through, with shuffles or 1×1 convs between them.

### Change a Scale into Coupling

**New task:** Use $x_1=z_1$, $x_2=2z_2+1$ through constant $s=\log2$, $t=1$.

**Picture and change:** Apply the triangular coupling and evaluate its explicit inverse.

**Check:** $(1,2)$ maps to $(1,5)$, determinant two; inverse second coordinate is $(5-1)/2=2$.

**Use when:** A structured invertible layer needs a tractable determinant without inverting its internal neural networks.

### Change the Required Fast Direction

**New task:** Prioritize density scoring in three coordinates over generation latency.

**Picture and change:** In MAF, known x prefixes support masked parallel inverse-coordinate evaluation.
Sampling must resolve the x prefixes sequentially; IAF reverses this directional convenience.

**Check:** Generating x1, x2, x3 takes three dependent coordinate stages in this ordering.

**Use when:** Density queries and sample generation have different serving priorities.

The map $x=z^2$ is not a bijection on the real line: positive and negative z collide.
Using only one inverse branch in the stated formula loses probability mass.

<!-- learning:core-end -->
<div data-learning-extension>

## Common flow families

| Family | Structure | Important limit |
|--------|-----------|-----------------|
| Coupling | Triangular Jacobian; NICE is an additive special case | Expressiveness needs suitable compositions |
| MAF/IAF | Autoregressive coordinate dependence | Fast direction differs |
| Continuous time | ODE and divergence integration | Solver/trace estimation introduces numerical or Monte Carlo error |
| Invertible channel mixing | Learned nonsingular linear transform | Retain determinant and conditioning checks |

## When to use flows in 2026

Compare density estimation, simulation-based conditional inference, forecasting, and generative tasks under their actual accuracy and compute criteria.
A high likelihood does not automatically identify in-distribution semantics or provide a reliable anomaly/OOD score.
Do not infer a universal image-quality or training-speed ranking from model-family names.

## Common pitfalls

- **Wrong determinant direction:** density evaluation adds inverse log determinant, equivalently subtracting forward log determinant.
- **Discrete observations:** dequantization models a continuous auxiliary distribution; relate its density objective to discrete likelihood explicitly.
- **Numerics:** finite precision, nearly singular transforms, and ODE tolerances can invalidate an unqualified "exact" claim.

## Related

- [Variational autoencoders](/concepts/variational-autoencoders/). Alternative latent-variable generative model.
- [Autoregressive vs. diffusion](/concepts/autoregressive-vs-diffusion/). Broader paradigm comparison.

</div>
