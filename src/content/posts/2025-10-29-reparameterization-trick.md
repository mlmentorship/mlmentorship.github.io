---
title: "Explain the reparameterization trick"
description: "How VAEs propagate gradients through a sampling step. The senior answer explains the why (you can't differentiate through a sample) and the how (move the randomness outside the parameters)."
date: "2025-10-29"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth and generative-model interviews.*

A standard generative-model question. The L4 candidate states the formula. The L6 candidate explains why naive sampling breaks gradients and how the trick fixes it.

## The problem

Suppose you want to train a model that samples a latent variable `z` from a distribution `q_phi(z | x)` and uses `z` to reconstruct `x`. Loss `L(theta, phi)` depends on `z`, which is a *sample*.

To train, you need `dL / dphi`. But `z = sample(q_phi)` is a stochastic operation; the gradient through a sample isn't well-defined in general.

## The trick

Reparameterize the sample:

```
z = mu_phi(x) + sigma_phi(x) * epsilon,  where epsilon ~ N(0, I)
```

The randomness now comes from `epsilon`, which doesn't depend on `phi`. The sample `z` is now a deterministic function of `(phi, epsilon)`. The gradient `dL / dphi` is computed straightforwardly via chain rule.

In short: instead of "sample `z` from `q_phi`," do "sample `epsilon` from a fixed distribution, then transform deterministically."

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
			<text class="viz-axis-label" x="18" y="27">1 · NAIVE SAMPLE: STOCHASTIC NODE BREAKS THE PATH</text>
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
	<figcaption><strong>Read it this way:</strong> follow solid arrows to sample in either panel. Then follow the dashed gradient backward. The naive graph meets a stochastic draw with no ordinary local derivative; the reparameterized graph holds the sampled epsilon fixed and differentiates through <code>z = mu + sigma * epsilon</code> all the way to <code>phi</code>. The distribution of <code>z</code> is unchanged. Original construction checked against <a href="https://arxiv.org/abs/1312.6114">Kingma and Welling's VAE formulation</a>.</figcaption>
</figure>

## Lower-variance gradients

This is the foundational trick for variational autoencoders (VAE). Without it, you'd have to use REINFORCE / score-function gradients, which have much higher variance and need many samples to be useful.

## What an L5 answer sounds like

> "When you sample `z = sample(q_phi(z | x))` and use `z` in a downstream loss, you can't backprop through the sampling because the operation is stochastic.
>
> The trick: rewrite the sample as a deterministic function of (parameters, noise), where the noise comes from a fixed distribution.
>
> For a Gaussian: `z = mu(x) + sigma(x) * epsilon`, `epsilon ~ N(0, I)`. Now `z` is differentiable w.r.t. `mu` and `sigma`, which are computed by the encoder network. Gradients flow through normally.
>
> Used in VAEs, in some RL algorithms (Gumbel-softmax for discrete actions is the discrete analog), in normalizing flows, in continuous-control policy gradient methods."

This is L5. Mechanism explained, examples given.

## What an L6 answer adds

> "...some additional points:
>
> **It only works for distributions you can reparameterize.** Gaussian, Laplace, exponential, uniform: easy. Discrete distributions: hard (requires Gumbel-softmax with continuous relaxation). General distributions: requires implicit differentiation tricks.
>
> **Variance is much lower than score-function gradients.** For a 1D Gaussian, reparameterization gradient variance scales with the Jacobian of the network; score-function variance scales with `1 / sigma^2` of the sample, which can be huge. Empirically, reparameterization needs 1-10 samples to estimate a useful gradient; REINFORCE often needs 1000+.
>
> **For discrete latents (e.g., latent-variable models with categorical z), Gumbel-softmax / concrete distribution** is the standard relaxation. Use a continuous relaxation that's differentiable; anneal the temperature toward zero to make samples nearly discrete. Trade-off: differentiable but biased.
>
> **In LLM-RL (RLHF, DPO), reparameterization isn't used** because text generation is discrete and Gumbel-softmax over a vocabulary of 50K tokens is impractical. RLHF uses score-function gradients (PPO), accepting the variance cost. DPO sidesteps this entirely with an analytical objective that doesn't need sampling at all."

## Tells that get you a strong-hire vote

- You explain **why** sampling breaks gradients before stating the trick.
- You give the **Gaussian formula** explicitly.
- You bring up **Gumbel-softmax** for discrete latents.
- You compare **variance** to score-function gradients.
- You mention **DPO sidesteps this** for LLM RL.

## Tells that get you down-leveled

- Stating the formula without explanation.
- Confusion about why naive sampling doesn't work.
- No knowledge of discrete-relaxation alternatives.
- Treating reparameterization as a VAE-only trick (it's broader).

## Common follow-up

"How does this apply to RL?"

The L6 answer:

> "Two cases. For *continuous-action policies* (e.g., robotic control with a Gaussian policy), reparameterize the action sample and backprop through the value function (this is the basic trick behind DDPG and SAC). For *discrete-action policies* (e.g., RL on discrete decisions), reparameterization doesn't directly apply; you use score-function gradients (REINFORCE, A2C, PPO) and accept the variance cost, sometimes mitigated by control variates and baselines. The 'continuous vs discrete' choice often dominates the algorithm choice in modern RL."

---

*Related: [RLHF and DPO](/concepts/rlhf-and-dpo/), [cross-entropy and softmax](/concepts/cross-entropy-softmax/), and [Bayesian versus frequentist](/questions/bayesian-vs-frequentist/).*
