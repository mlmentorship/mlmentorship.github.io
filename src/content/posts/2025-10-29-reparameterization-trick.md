---
title: "Explain the reparameterization trick"
description: "Differentiate a smooth parameterized sample path with fixed base noise, then distinguish its gradient estimate from the expectation gradient."
date: "2025-10-29"
draft: false
tags: ["questions"]
category: "questions"
---


Derive a gradient estimator for an expected loss whose sampled latent variable depends on trainable distribution parameters.

## Problem

**Context.** A Gaussian latent variable has trainable mean and positive standard deviation, and a differentiable loss depends on its sample.
Training requires the derivative of the expected loss, not merely a derivative of the density.

**Task.** Express sampling through parameter-independent noise, derive the pathwise gradient, and explain how Monte Carlo estimation uses it.
Compare what can change for discrete samples or nondifferentiable transformations.

**Constraints.** State conditions permitting differentiation under the expectation.
Hold base noise fixed during a sample's derivative; distinguish that sample estimate from the exact expectation gradient.
Do not assume every distribution or objective supports the same transformation.

## The problem

Let $F(\phi)=E_{q_\phi(z\mid x)}[f(z)]$ for fixed x.
A sampler returning a detached draw supplies no ordinary local autodiff path to its distribution parameters.
This is a property of the representation and estimator, not proof that every stochastic expectation is nondifferentiable.

## The trick

Reparameterize the sample:

```
z = mu_phi(x) + sigma_phi(x) * epsilon,  where epsilon ~ N(0, I)
```

For positive standard deviation sigma, the Gaussian sample has the desired distribution while its transform is differentiable.
When differentiation and expectation can be interchanged,

$$
\nabla_\phi F=E_\epsilon[\nabla_z f(z)\,\nabla_\phi z].
$$

If f also depends explicitly on phi, include that additional derivative.
For $\mu=1$, $\sigma=2$, $\epsilon=0.5$, and $f(z)=(z-3)^2$, the sample is two and loss one.
The local derivative is $2(z-3)=-2$, so $df/d\mu=-2$ and $df/d\sigma=-1$ for this fixed draw.

The expected loss instead is $F=(\mu-3)^2+\sigma^2=8$, with derivatives $(-4,4)$.
Noise values minus one and plus one give sample gradients $(-8,8)$ and $(0,0)$, averaging to $(-4,4)$.
That two-point check matches this quadratic's moments; it is not Gaussian sampling or a general integration rule.

<!-- visual:reparameterization-gradient-path -->
<figure class="learning-figure" aria-labelledby="reparameterization-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="reparameterization-visual-title">Move randomness outside the parameterized path so ordinary backpropagation can reach the encoder.</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 450" role="img" aria-labelledby="reparameterization-svg-title reparameterization-svg-desc">
			<title id="reparameterization-svg-title">Naive sampling compared with a reparameterized Gaussian sample</title>
			<desc id="reparameterization-svg-desc">Two stacked computation graphs use solid arrows for the forward pass and dashed arrows for gradients. In the naive graph, encoder parameters phi define q phi of z given x, which produces a random draw z and then a loss. The reverse gradient reaches z but an X marks that ordinary backpropagation stops at the stochastic draw. In the reparameterized graph, the encoder produces mu and sigma while epsilon is sampled independently from a standard normal. Both feed the deterministic transform z equals mu plus sigma times epsilon, then the loss. Dashed arrows form an unbroken reverse path from the loss through z, mu and sigma, and back to phi. No gradient is needed through epsilon.</desc>
			<defs>
				<marker id="reparameterization-forward-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
				<marker id="reparameterization-backward-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="5" y="5" width="350" height="170" rx="6"></rect>
			<text class="viz-axis-label" x="18" y="27">1 · DETACHED DRAW: NO LOCAL AUTODIFF PATH</text>
			<rect class="viz-node viz-node--input" x="16" y="54" width="74" height="48" rx="5"></rect>
			<text class="viz-node-label" x="53" y="75">φ</text><text class="viz-node-value" x="53" y="92">encoder params</text>
			<rect class="viz-node" x="111" y="54" width="82" height="48" rx="5"></rect>
			<text class="viz-node-label" x="152" y="75">q<tspan baseline-shift="sub" font-size="9">φ</tspan>(z|x)</text><text class="viz-node-value" x="152" y="92">distribution</text>
			<rect class="viz-node viz-node--focus" x="214" y="54" width="58" height="48" rx="5"></rect>
			<text class="viz-node-label" x="243" y="75">z</text><text class="viz-node-value" x="243" y="92">random draw</text>
			<rect class="viz-node viz-node--output" x="293" y="54" width="51" height="48" rx="5"></rect>
			<text class="viz-node-label" x="318.5" y="75">L(z)</text><text class="viz-node-value" x="318.5" y="92">loss</text>
			<path d="M90 78H107" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#reparameterization-forward-arrow)"></path>
			<path d="M193 78H210" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#reparameterization-forward-arrow)"></path>
			<path d="M272 78H289" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#reparameterization-forward-arrow)"></path>
			<path d="M318 108V127H247V106" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#reparameterization-backward-arrow)"></path>
			<path d="M200 116L212 128M212 116L200 128" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5"></path>
			<text class="viz-gradient-label" x="282" y="143" style="font-size:11px">∂L/∂z</text>
			<text class="viz-edge-label" x="154" y="153" style="font-size:11px">X · ordinary backprop stops at the draw</text>
			<rect class="viz-plot-bg" x="5" y="190" width="350" height="255" rx="6"></rect>
			<text class="viz-axis-label" x="18" y="212">2 · REPARAMETERIZED: SAME DISTRIBUTION, OPEN PATH</text>
			<rect class="viz-node viz-node--input" x="18" y="239" width="88" height="50" rx="5"></rect>
			<text class="viz-node-label" x="62" y="260">encoder φ</text><text class="viz-node-value" x="62" y="278">input x</text>
			<rect class="viz-node" x="136" y="239" width="88" height="50" rx="5"></rect>
			<text class="viz-node-label" x="180" y="260">μ<tspan baseline-shift="sub" font-size="9">φ</tspan>, σ<tspan baseline-shift="sub" font-size="9">φ</tspan></text><text class="viz-node-value" x="180" y="278">differentiable</text>
			<rect class="viz-node viz-node--input" x="18" y="329" width="88" height="50" rx="5"></rect>
			<text class="viz-node-label" x="62" y="350">ε ∼ N(0,I)</text><text class="viz-node-value" x="62" y="368">independent of φ</text>
			<rect class="viz-node viz-node--focus" x="136" y="329" width="88" height="50" rx="5"></rect>
			<text class="viz-node-label" x="180" y="350">z = μ + σε</text><text class="viz-node-value" x="180" y="368">deterministic</text>
			<rect class="viz-node viz-node--output" x="256" y="329" width="86" height="50" rx="5"></rect>
			<text class="viz-node-label" x="299" y="350">L(z)</text><text class="viz-node-value" x="299" y="368">loss</text>
			<path d="M106 258H132" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#reparameterization-forward-arrow)"></path>
			<path d="M174 289V325" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#reparameterization-forward-arrow)"></path>
			<path d="M106 354H132" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#reparameterization-forward-arrow)"></path>
			<path d="M224 354H252" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#reparameterization-forward-arrow)"></path>
			<path d="M299 379V395H180V383" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#reparameterization-backward-arrow)"></path>
			<path d="M186 329V293" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#reparameterization-backward-arrow)"></path>
			<path d="M136 270H110" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#reparameterization-backward-arrow)"></path>
			<text class="viz-gradient-label" x="180" y="414" style="font-size:11px">chain rule: ∂L/∂z → ∂L/∂(μ,σ) → ∂L/∂φ</text>
			<text class="viz-edge-label" x="180" y="433" style="font-size:11px">epsilon supplies a value; no derivative through its draw is needed</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> a detached draw supplies no ordinary local autodiff path. The lower graph holds base noise fixed and differentiates its deterministic Gaussian transform. It preserves the sampling distribution while changing the available gradient path. Original construction follows <a href="https://arxiv.org/abs/1312.6114">Kingma and Welling</a>.</figcaption>
</figure>

## Lower-variance gradients

Pathwise estimates can reduce variance relative to score-function estimates, but no distribution-free variance ordering or sample-count guarantee holds.
Score-function methods, baselines, analytic terms, and other estimators remain useful under their respective conditions.

## What an L5 answer sounds like

A complete explanation states the noise law, transform, held-fixed sample, distributional identity, and interchange assumptions before applying the chain rule.

## What an L6 answer adds

### Change the Encoder's Scale Output

**New task:** The encoder emits log variance r instead of standard deviation sigma.

**Picture and change:** Use $\sigma=\exp(r/2)$ and retain the same epsilon, chaining through $d\sigma/dr=\sigma/2$.

**Check:** At $r=\log4$, sigma is two; the reference sample's $df/dr=(-1)(2/2)=-1$.
Using variance four as the standard deviation changes the distribution and derivative.

**Use when:** An unconstrained encoder output needs conversion to positive scale.

### Change the Latent Support

**New task:** Replace the Gaussian latent with Bernoulli probability p.

**Picture and change:** Thresholding uniform noise gives $z=1[u<p]$, whose sample derivative is zero almost everywhere.

**Check:** Nevertheless, $E[z]=p$ has derivative one; averaging those zero derivatives fails to recover it.

**Use when:** Discrete support breaks the smooth sample-path argument.
Use a suitable score-function estimator or a justified relaxation, not a Gaussian substitution.

For framework use, keep mu/log-variance/noise axes aligned, commonly `[B,D]`, and preserve the draw's differentiable transform.
PyTorch Gaussian `rsample()` supplies that path; `sample()` ordinarily does not.
K samples require $O(KBD)$ noise/transform work plus downstream computation.
Detaching z blocks the path; fresh noise invalidates the finite-difference comparison.

<!-- learning:core-end -->
<div data-learning-extension>

## Tells that get you a strong-hire vote

Explain the sample derivative, expected derivative, and their different numerical values rather than presenting a level speech.

## Tells that get you down-leveled

Avoid claiming that all stochastic nodes are intrinsically nondifferentiable or that pathwise gradients always have lower variance.
Gumbel-softmax differentiates a relaxed objective; annealing temperature does not guarantee unbiased discrete-objective gradients.

## Common follow-up

SAC can differentiate a reparameterized continuous-action sample through a learned critic under its objective.
DDPG instead uses a deterministic policy gradient; it is not the same Gaussian sampling construction.
Discrete-action policies can use score-function estimators and baselines; algorithm choice also depends on data, constraints, and learning assumptions.
DPO uses a preference-data objective rather than backpropagating through generated discrete tokens; this does not make all data collection unnecessary.

---

*Related: [RLHF and DPO](/concepts/rlhf-and-dpo/), [cross-entropy and softmax](/concepts/cross-entropy-softmax/), and [Bayesian versus frequentist](/questions/bayesian-vs-frequentist/).*

</div>
