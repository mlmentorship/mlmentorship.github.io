---
title: "KL divergence"
description: "Asymmetric distance between probability distributions. Cross-entropy minus entropy. The mathematical glue holding most of probabilistic ML together."
date: "2025-10-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

For probability distributions $p$ and $q$ over the same space:

$$
\mathrm{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \quad (\text{or } \int p \log(p/q)\, dx \text{ for continuous}).
$$

It's the expected log-ratio of $p$ to $q$ under $p$. Measuring how much information is lost using $q$ to encode samples from $p$.

KL divergence is the fundamental object of statistical learning. It connects:

- Maximum likelihood (minimizing $\mathrm{KL}(\hat p_\text{data} \| p_\theta)$).
- Variational inference (minimizing $\mathrm{KL}(q_\phi \| p)$).
- Cross-entropy loss = entropy of data + KL.
- Information bottleneck and mutual information.
- Policy gradient methods in RL (TRPO, PPO use KL constraints).
- Knowledge distillation (student matches teacher distribution via KL).

## Properties

- **Non-negative**: $\mathrm{KL}(p \| q) \ge 0$, with equality iff $p = q$ (Gibbs' inequality).
- **Asymmetric**: $\mathrm{KL}(p \| q) \ne \mathrm{KL}(q \| p)$ in general. Choose direction based on whether you are "fitting $q$ to $p$" or vice versa.
- **Not a metric**: no triangle inequality, not symmetric.
- **Infinite if $q(x) = 0$ where $p(x) > 0$**: $q$ must cover the support of $p$.
- **Information-theoretic**: equals expected extra bits (or nats) per sample needed to encode $p$ using a code optimized for $q$.

## Forward vs. reverse KL

The asymmetry matters in practice. For approximating $p$ with $q$:

- **Forward KL**, $\mathrm{KL}(p \| q)$: penalizes $q$ for missing modes of $p$ ("**mass-covering**"; under a restricted unimodal family, the result often sits between modes). Used in standard MLE.
- **Reverse KL**, $\mathrm{KL}(q \| p)$: penalizes $q$ for placing mass where $p$ has none ("**mode-seeking**". $q$ collapses to one mode). Used in variational inference.

For a multimodal $p$ and a restricted approximation family, forward KL often gives a broad average while reverse KL can pick one mode.

<!-- visual:kl-direction-candidate-ranking -->
<figure class="learning-figure plot-panel" aria-labelledby="kl-direction-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="kl-direction-title">Why can swapping the KL arguments change which approximation wins?</p>
	<svg viewBox="0 0 360 500" role="img" aria-labelledby="kl-direction-svg-title kl-direction-svg-desc">
		<title id="kl-direction-svg-title">Forward and reverse KL rank two approximations differently</title>
		<desc id="kl-direction-svg-desc">A three-outcome target distribution p has probabilities 0.49, 0.02, and 0.49, forming two modes. A broad candidate q cover has probabilities 0.34, 0.32, and 0.34. A one-mode candidate q left has probabilities 0.96, 0.02, and 0.02. Forward KL from p to q is 0.30 nats for the broad candidate and 1.24 for the one-mode candidate, so it selects broad coverage. Reverse KL from q to p is 0.64 nats for the broad candidate and 0.58 for the one-mode candidate, so it selects one mode. The example compares only these fixed candidates.</desc>
		<rect class="viz-plot-bg" x="8" y="8" width="344" height="118" rx="6"></rect>
		<text class="viz-axis-label" x="18" y="27">TARGET p · TWO MODES</text>
		<path class="viz-axis" d="M44 110H316"></path>
		<rect class="viz-node" x="70" y="73.25" width="40" height="36.75"></rect>
		<rect class="viz-node" x="155" y="108.5" width="40" height="1.5"></rect>
		<rect class="viz-node" x="240" y="73.25" width="40" height="36.75"></rect>
		<text class="viz-callout" x="90" y="67" text-anchor="middle">0.49</text>
		<text class="viz-callout" x="175" y="101" text-anchor="middle">0.02</text>
		<text class="viz-callout" x="260" y="67" text-anchor="middle">0.49</text>
		<text class="viz-label" x="90" y="122" text-anchor="middle">left</text>
		<text class="viz-label" x="175" y="122" text-anchor="middle">middle</text>
		<text class="viz-label" x="260" y="122" text-anchor="middle">right</text>
		<rect class="viz-plot-bg" x="8" y="136" width="344" height="118" rx="6"></rect>
		<text class="viz-axis-label" x="18" y="155">CANDIDATE A · q cover · BROAD</text>
		<path class="viz-axis" d="M44 238H316"></path>
		<rect class="viz-node viz-node--focus" x="70" y="212.5" width="40" height="25.5"></rect>
		<rect class="viz-node viz-node--focus" x="155" y="214" width="40" height="24"></rect>
		<rect class="viz-node viz-node--focus" x="240" y="212.5" width="40" height="25.5"></rect>
		<text class="viz-callout" x="90" y="206" text-anchor="middle">0.34</text>
		<text class="viz-callout" x="175" y="207.5" text-anchor="middle">0.32</text>
		<text class="viz-callout" x="260" y="206" text-anchor="middle">0.34</text>
		<text class="viz-label" x="18" y="247">Places mass across both target modes and the low-density middle.</text>
		<rect class="viz-plot-bg" x="8" y="264" width="344" height="118" rx="6" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:7 4"></rect>
		<text class="viz-axis-label" x="18" y="283">CANDIDATE B · q left · ONE MODE · DASHED PANEL</text>
		<path class="viz-axis" d="M44 360H316"></path>
		<rect class="viz-node viz-node--focus" x="70" y="288" width="40" height="72" style="stroke-dasharray:7 4"></rect>
		<rect class="viz-node viz-node--focus" x="155" y="358.5" width="40" height="1.5" style="stroke-dasharray:7 4"></rect>
		<rect class="viz-node viz-node--focus" x="240" y="358.5" width="40" height="1.5" style="stroke-dasharray:7 4"></rect>
		<text class="viz-callout" x="90" y="304" text-anchor="middle">0.96</text>
		<text class="viz-callout" x="175" y="357" text-anchor="middle">0.02</text>
		<text class="viz-callout" x="260" y="357" text-anchor="middle">0.02</text>
		<text class="viz-label" x="18" y="375">Concentrates on the left target mode and nearly ignores the right.</text>
		<rect class="viz-node viz-node--output" x="8" y="392" width="344" height="100" rx="6"></rect>
		<text class="viz-axis-label" x="18" y="411">DIRECTION · NATURAL-LOG UNITS</text>
		<text class="viz-label" x="245" y="411" text-anchor="middle">BROAD</text>
		<text class="viz-label" x="318" y="411" text-anchor="middle">ONE MODE</text>
		<text class="viz-callout" x="18" y="437">KL(p || q) · p weights misses</text>
		<text class="viz-callout" x="245" y="437" text-anchor="middle">0.30 · PICK</text>
		<text class="viz-callout" x="318" y="437" text-anchor="middle">1.24</text>
		<path class="viz-gridline" d="M18 447H342"></path>
		<text class="viz-callout" x="18" y="467">KL(q || p) · q weights extras</text>
		<text class="viz-callout" x="245" y="467" text-anchor="middle">0.64</text>
		<text class="viz-callout" x="318" y="467" text-anchor="middle">0.58 · PICK</text>
		<text class="viz-label" x="18" y="485">Same target and candidates; only the expectation changes.</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> hold the target and both candidates fixed, then read each score row from left to right. In <code>KL(p || q)</code>, target mass weights the penalty, so nearly missing the right mode makes the one-mode candidate expensive. In <code>KL(q || p)</code>, candidate mass weights the penalty, so the broad candidate pays for its extra middle mass while the one-mode candidate scarcely samples the missed mode. This original three-outcome construction illustrates a restricted choice between two candidates, not a universal optimizer. Values are in nats and were checked against the KL definition in <a href="https://doi.org/10.1214/aoms/1177729694">Kullback and Leibler (1951)</a> and the inclusive/exclusive analysis in <a href="https://www.microsoft.com/en-us/research/publication/divergence-measures-and-message-passing/">Minka (2005)</a>.</figcaption>
</figure>

## Connection to cross-entropy

For empirical distribution $\hat p$ over a finite dataset:

$$
H(\hat p, p_\theta) = -\mathbb{E}_{\hat p}[\log p_\theta] = H(\hat p) + \mathrm{KL}(\hat p \,\|\, p_\theta).
$$

Cross-entropy = entropy + KL. Since entropy of the data doesn't depend on $\theta$, minimizing cross-entropy = minimizing KL = MLE. This is why "the loss is cross-entropy" and "we're minimizing KL to the data" are the same statement.

## Common usage in ML

| Use case | Direction |
|---------|-----------|
| Classification cross-entropy loss | Forward $\mathrm{KL}(\text{data} \| \text{model})$ |
| Variational inference (ELBO) | Reverse $\mathrm{KL}(q \| p)$ |
| RLHF / PPO penalty | Reverse $\mathrm{KL}(\pi_\theta \| \pi_\text{ref})$. Keep new policy close to reference |
| Knowledge distillation | Forward $\mathrm{KL}(\text{teacher} \| \text{student})$ with temperature |
| t-SNE | Forward $\mathrm{KL}(P \| Q)$ on pairwise similarities |

## Common pitfalls

- **Computing KL between distributions with different supports.** If $p(x) > 0$ and $q(x) = 0$, KL is $+\infty$.
- **Confusing JS divergence (symmetric) with KL.** GANs originally used JS; modern variants (Wasserstein) avoid both.
- **Forgetting the asymmetry direction.** Forward and reverse KL produce qualitatively different optimizers.
- **Using KL on samples without density estimates.** KL is between distributions, not between sample sets; sample-based estimators are noisy and biased.

*Related: [entropy, mutual information, and information gain](/concepts/entropy-mutual-information/), [cross-entropy and softmax](/concepts/cross-entropy-softmax/), and [variational autoencoders](/concepts/variational-autoencoders/).*
