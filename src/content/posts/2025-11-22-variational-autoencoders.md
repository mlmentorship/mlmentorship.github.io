---
title: "Variational autoencoders: calculate the bound and its gap"
description: "Evaluate a complete Gaussian ELBO, improve the approximate posterior, and test why changing the KL weight changes the guarantee."
date: "2025-11-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A variational autoencoder fits a latent-variable model using approximate posterior inference.
Its encoder estimates latent uncertainty for observed data; its decoder specifies an observation distribution, not merely a reconstructed mean.

**Task:** observe $x=2$ under $p(z)=\mathcal N(0,1)$ and $p(x\mid z)=\mathcal N(z,1)$.
Use approximate posterior $q(z\mid x)=\mathcal N(1,1/4)$, where the second argument denotes variance.
Compute the training bound and compare it with the exact log-likelihood.
This scalar teaching model is solvable; neural models often make the marginal integral difficult.

## The model

Generally, $p_\theta(x,z)=p(z)p_\theta(x\mid z)$ and the encoder is $q_\phi(z\mid x)$.
Parameters $\theta$ define the generative model; $\phi$ defines its approximate inference network.
One encoder shares inference parameters across observations: amortization does not make each inferred posterior exact.

Read the training lane and generation lane as different computations.

<!-- visual:vae-training-and-generation-paths -->
<figure class="learning-figure plot-panel" aria-labelledby="vae-paths-title">
	<p class="visual-kicker">One model, two execution paths</p>
	<p class="visual-title" id="vae-paths-title">Generation samples the prior without an observed input or encoder</p><div class="visual-argument" data-visual-argument="vae-training-and-generation-paths" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 194" role="img" aria-label="Generation has no observed input to encode. Training infers z from observed x through the encoder. Generation instead samples z from the prior and decodes it without q(z|x)."><title>Generation has no observed input to encode</title><desc>Training infers z from observed x through the encoder. Generation instead samples z from the prior and decodes it without q(z|x).</desc><text x="133.6" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="133.6" dy="0">observed</tspan><tspan x="133.6" dy="18">x</tspan></text><text x="216.16" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="216.16" dy="0">latent</tspan><tspan x="216.16" dy="18">source</tspan></text><text x="298.72" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="298.72" dy="0">decoder</tspan></text><text x="80.32" y="72" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="80.32" dy="0">train</tspan></text><g data-proof-status="present"><rect data-proof-cell="0,0" x="92.32" y="54" width="77.56" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><path d="M124.1 72l5 5 10-12" style="fill:none;stroke:var(--c-text);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="131.1" y="100" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="131.1" dy="0">yes</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="174.88" y="54" width="77.56" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="213.66" y="89" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="213.66" dy="0">q(z|x)</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,2" x="257.44" y="54" width="77.56" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="296.22" y="89" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="296.22" dy="0">decode</tspan></text></g><text x="80.32" y="136" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="80.32" dy="0">generate</tspan></text><g data-proof-status="absent"><rect data-proof-cell="1,0" x="92.32" y="118" width="77.56" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="131.1" cy="136" r="9" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M126.1 136L136.1 136" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="131.1" y="164" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="131.1" dy="0">no</tspan></text><path data-proof-warning="1,0" d="M96.32 126v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="174.88" y="118" width="77.56" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="213.66" y="153" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="213.66" dy="0">prior</tspan></text><path d="M179.88 171L247.44 171" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="value"><rect data-proof-cell="1,2" x="257.44" y="118" width="77.56" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="296.22" y="153" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="296.22" dy="0">decode</tspan></text><path d="M262.44 171L330 171" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 490" role="img" aria-labelledby="vae-paths-svg-title vae-paths-svg-desc">
		<title id="vae-paths-svg-title">VAE training and generation paths</title>
		<desc id="vae-paths-svg-desc">In the upper training panel, observed x enters the encoder, which produces approximate posterior q phi. A reparameterized latent z flows into the decoder and the reconstruction term. Beside that path, the KL term compares q phi with prior p of z. In the lower generation panel, a dashed path samples z directly from the prior, passes it through the decoder, and produces a distribution for a new observation. The encoder and training input do not participate in generation.</desc>
		<defs>
			<marker id="vae-solid-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
			<marker id="vae-dashed-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="5" y="5" width="350" height="324" rx="6"></rect>
		<text class="viz-axis-label" x="18" y="28">TRAIN: INFER z FROM THIS x</text>
		<rect class="viz-node viz-node--input" x="18" y="48" width="72" height="46" rx="5"></rect>
		<text class="viz-node-label" x="54" y="68">x</text>
		<text class="viz-node-value" x="54" y="85">observed</text>
		<rect class="viz-node" x="111" y="48" width="82" height="46" rx="5"></rect>
		<text class="viz-node-label" x="152" y="68">encoder</text>
		<text class="viz-node-value" x="152" y="85">parameters φ</text>
		<rect class="viz-node viz-node--focus" x="214" y="48" width="128" height="46" rx="5"></rect>
		<text class="viz-node-label" x="278" y="68">q<tspan baseline-shift="sub" font-size="9">φ</tspan>(z|x)</text>
		<text class="viz-node-value" x="278" y="85">μ<tspan baseline-shift="sub" font-size="8">φ</tspan>(x), σ<tspan baseline-shift="sub" font-size="8">φ</tspan>(x)</text>
		<path d="M90 71H107" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<path d="M193 71H210" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<rect class="viz-node" x="18" y="132" width="82" height="46" rx="5" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-node-label" x="59" y="152">p(z)</text>
		<text class="viz-node-value" x="59" y="169">prior N(0,I)</text>
		<rect class="viz-node viz-node--focus" x="118" y="132" width="76" height="46" rx="5" style="stroke-dasharray:6 4"></rect>
		<text class="viz-node-label" x="156" y="152">KL</text>
		<text class="viz-node-value" x="156" y="169">regularizer</text>
		<rect class="viz-node viz-node--focus" x="230" y="132" width="96" height="46" rx="5"></rect>
		<text class="viz-node-label" x="278" y="152">z = μ + σε</text>
		<text class="viz-node-value" x="278" y="169">ε ∼ N(0,I)</text>
		<path d="M278 94V128" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<path d="M214 78H205V126H194" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.8;marker-end:url(#vae-solid-arrow)"></path>
		<path d="M100 155H114" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.8;marker-end:url(#vae-solid-arrow)"></path>
		<text class="viz-label" x="156" y="118" text-anchor="middle">compare q<tspan baseline-shift="sub" font-size="8">φ</tspan> to p</text>
		<rect class="viz-node" x="230" y="216" width="96" height="46" rx="5"></rect>
		<text class="viz-node-label" x="278" y="236">decoder</text>
		<text class="viz-node-value" x="278" y="253">parameters θ</text>
		<path d="M278 178V212" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<rect class="viz-node viz-node--output" x="145" y="279" width="197" height="28" rx="14"></rect>
		<text class="viz-node-value" x="243.5" y="297">reconstruction: log p<tspan baseline-shift="sub" font-size="8">θ</tspan>(x|z)</text>
		<path d="M278 262V275" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vae-solid-arrow)"></path>
		<path d="M90 82V293H141" style="fill:none;stroke:var(--viz-edge);stroke-width:1.4;stroke-dasharray:4 4;marker-end:url(#vae-solid-arrow)"></path>
		<text class="viz-label" x="25" y="200">target x</text>
		<text class="viz-callout" x="18" y="322">ELBO = reconstruction - KL</text>
		<rect class="viz-plot-bg" x="5" y="335" width="350" height="150" rx="6"></rect>
		<text class="viz-axis-label" x="18" y="358">GENERATE: SAMPLE z WITHOUT AN INPUT x</text>
		<rect class="viz-node" x="18" y="384" width="82" height="46" rx="5" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-node-label" x="59" y="404">p(z)</text>
		<text class="viz-node-value" x="59" y="421">sample z</text>
		<rect class="viz-node" x="139" y="384" width="82" height="46" rx="5"></rect>
		<text class="viz-node-label" x="180" y="404">decoder</text>
		<text class="viz-node-value" x="180" y="421">same θ</text>
		<rect class="viz-node viz-node--output" x="260" y="384" width="82" height="46" rx="5"></rect>
		<text class="viz-node-label" x="301" y="404">p<tspan baseline-shift="sub" font-size="9">θ</tspan>(x|z)</text>
		<text class="viz-node-value" x="301" y="421">new x</text>
		<path d="M100 407H135" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#vae-dashed-arrow)"></path>
		<path d="M221 407H256" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#vae-dashed-arrow)"></path>
		<text class="viz-label" x="180" y="461" text-anchor="middle">No observed x, encoder, q<tspan baseline-shift="sub" font-size="8">φ</tspan>, or KL step.</text>
	</svg>
	<div class="visual-derivation-caption">observed <var>x</var> enters the encoder on the solid training path. A posterior sample reaches the decoder; reconstruction scores the conditional density of that same <var>x</var>. The KL term compares the approximate posterior with the prior. The dashed generation path instead samples the prior and uses the decoder without an observed <var>x</var>. Sources: <a href="https://arxiv.org/abs/1312.6114">Kingma and Welling</a>, <a href="https://arxiv.org/abs/1401.4082">Rezende et al.</a>, and <a href="https://arxiv.org/abs/1906.02691">the VAE tutorial</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="vae-training-and-generation-paths"><strong>Check:</strong> Why does generation need neither an observed x nor q(z|x), unlike training?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="vae-training-and-generation-paths">Training infers z from observed x through the encoder. Generation instead samples z from the prior and decodes it without q(z|x).</span></figcaption>
</figure>

## The ELBO

For these everywhere-positive Gaussian densities, insert $q$ and apply Jensen's inequality:

$$
\log p_\theta(x)=\log\mathbb E_q\left[\frac{p_\theta(x,z)}{q(z\mid x)}\right]
\ge\mathbb E_q[\log p_\theta(x,z)-\log q(z\mid x)]
=\mathbb E_q[\log p_\theta(x\mid z)]-\operatorname{KL}(q\|p(z)).
$$

Call the final expression $\mathcal L$, the evidence lower bound, or ELBO.
Bayes' rule also gives the exact gap identity:

$$
\log p_\theta(x)=\mathcal L+
\operatorname{KL}\bigl(q_\phi(z\mid x)\|p_\theta(z\mid x)\bigr).
$$

The penalty inside $\mathcal L$ compares $q$ with the prior; the gap compares $q$ with the model posterior.
These are different distributions.

For the fixture, $\mathbb E_q[(2-z)^2]=(2-1)^2+1/4=1.25$.
For a Gaussian encoder with mean $\mu$ and variance $s^2$, the prior KL is $(\mu^2+s^2-1-\log s^2)/2$.

| Quantity, natural-log units | Calculation | Value |
| --- | --- | --- |
| Expected reconstruction log density | $-\frac12\log(2\pi)-0.625$ | $-1.54394$ |
| Prior KL | $(1+1/4-1-\log(1/4))/2$ | $0.81815$ |
| ELBO | Reconstruction minus prior KL | $-2.36209$ |
| Exact log-likelihood, $x\sim\mathcal N(0,2)$ | $-\frac12\log(4\pi)-1$ | $-2.26551$ |

The exact posterior is $\mathcal N(1,1/2)$.
Directly computing its KL from $q$ gives $(1/2-1-\log(1/2))/2=0.09657$, exactly the likelihood-minus-bound gap.

### Improve inference variance

**New task:** change $q$'s variance from $1/4$ to the exact posterior variance $1/2$.
Update both reconstruction and prior KL, not just the penalty.
**Check:** reconstruction becomes $-1.66894$, prior KL $0.59657$, and ELBO $-2.26551$.
The posterior gap is zero.
**Use when:** the inference family can better represent the model posterior.

### Change the KL weight

**New task:** keep the original $q$ but optimize reconstruction minus $\beta$ times prior KL, with $\beta=1/2$.
**Check:** the objective becomes $-1.95301$, exceeding the exact log-likelihood $-2.26551$.
It is not the ordinary ELBO; its usual lower-bound guarantee cannot be asserted.
**Use when:** deliberately changing a representation or rate-distortion tradeoff rather than claiming the same likelihood objective.

## The reparameterization trick

Write $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$, with parameter-independent $\epsilon\sim\mathcal N(0,I)$.
For log variance $\ell$, use $\sigma=\exp(\ell/2)$, not $\exp\ell$.
The randomness remains; its sampling law no longer depends on $\phi$.
See the [pathwise-gradient derivation](/questions/reparameterization-trick/) for conditions and alternatives.

**Reconstruct:** which observed $x$ enters the encoder during unconditional generation?

<details><summary>Answer</summary>

None. Sample $z$ from the prior, then sample or summarize the decoder distribution $p_\theta(x\mid z)$.
Reconstruction instead conditions on a supplied observation through the encoder.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## What VAEs are good and bad at

Continuous latent models can support interpolation and learned compression, including latent representations used by downstream diffusion models.
Their usefulness depends on the learned distribution and reconstruction contract, not continuity alone.
Gaussian squared error corresponds to a specified decoder variance; it is not a universal reconstruction likelihood.
Blur depends on that likelihood, uncertainty, architecture, and optimization, not simply the VAE name.
No universal stability or sample-quality ranking follows against GANs or diffusion.

## Common pitfalls

Posterior collapse means inferred latents carry little observation-specific information and the decoder can ignore them.
An expressive autoregressive decoder can still produce varied outputs; collapse does not imply a constant image.
Annealing, free bits, decoder changes, and inference improvements alter different parts of training and offer no universal cure.
Not using a pathwise estimator does not uniquely force REINFORCE; other estimators depend on the latent family.
Check likelihood sums, minibatch reductions, and KL units before interpreting their relative weight.

## Variants

- **Beta-VAE:** changes the KL weight; disentanglement is not guaranteed by that coefficient alone.
- **VQ-VAE:** introduces discrete codebooks and a different training construction, not Gaussian reparameterization unchanged.
- **IWAE:** uses an importance-weighted objective with its own support and sampling conditions.
- **NVAE and hierarchical VAEs:** introduce multiple conditional latent levels, retaining a distinct factorization.

The [IWAE bound](https://arxiv.org/abs/1509.00519) is an expectation, not an exact likelihood correction.
A single Monte Carlo realization need not lie below log-likelihood.

## Related

- [Reparameterization trick](/questions/reparameterization-trick/). The gradient enabler.
- [Autoregressive vs. diffusion](/concepts/autoregressive-vs-diffusion/). Alternative generative paradigms.

</div>
