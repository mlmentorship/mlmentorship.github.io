---
title: "KL divergence: which distribution weights the error?"
description: "Score fixed candidates in both directions, derive the cross-entropy identity, and test support and approximation-family changes."
date: "2025-10-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

KL divergence averages a log probability ratio under its first distribution.
Swapping the arguments changes that weighting, so the same candidates can receive opposite rankings.
Start with [expected surprisal](/concepts/entropy-mutual-information/), not a distance analogy.

**Task:** compare only two candidates for a three-outcome target:

$$
p=(0.49,0.02,0.49),\quad
q_{cover}=(0.34,0.32,0.34),\quad
q_{left}=(0.96,0.02,0.02).
$$

Each vector sums to one.
Use natural logarithms, giving scores in nats:

$$
\operatorname{KL}(p\|q)=\mathbb E_p\left[\log\frac{p(X)}{q(X)}\right]
=\sum_x p(x)\log\frac{p(x)}{q(x)}.
$$

## Properties

Terms with $p(x)=0$ contribute zero, including when $q(x)=0$.
If $p(x)>0$ but $q(x)=0$, the divergence is infinite.
Otherwise finite probability vectors give a nonnegative score, zero exactly when the distributions agree.
KL is not symmetric and does not satisfy metric guarantees such as the triangle inequality.

## Forward vs. reverse KL

Hold the target and both candidates fixed while changing only the sampling weights.

<!-- visual:kl-direction-candidate-ranking -->
<figure class="learning-figure plot-panel" aria-labelledby="kl-direction-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="kl-direction-title">Changing which distribution does the averaging reverses this candidate ranking</p>
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
	<figcaption><strong>Read it this way:</strong> hold the target and both candidates fixed, then compare the score rows. Forward KL weights errors by target mass; nearly missing the right mode is expensive. Reverse KL weights errors by candidate mass; the broad candidate pays for its extra middle mass. This is a restricted choice between two candidates, not a universal optimizer. Values are in nats, following <a href="https://doi.org/10.1214/aoms/1177729694">Kullback and Leibler (1951)</a>. For divergence direction, see <a href="https://www.microsoft.com/en-us/research/publication/divergence-measures-and-message-passing/">Minka (2005)</a>. <span class="visual-reader-check" data-reader-check="kl-direction-candidate-ranking"><strong>Check:</strong> Why does changing the expectation's weights reverse the preferred candidate without changing either candidate distribution?</span></figcaption>
</figure>

## Connection to cross-entropy

Combining equal terms gives four complete expectations:

| Expectation | Weighted log terms | Score, nats |
| --- | --- | --- |
| $\operatorname{KL}(p\|q_{cover})$ | $0.98\log(0.49/0.34)+0.02\log(0.02/0.32)$ | $0.302699$ |
| $\operatorname{KL}(p\|q_{left})$ | $0.49\log(0.49/0.96)+0.49\log(0.49/0.02)$ | $1.237811$ |
| $\operatorname{KL}(q_{cover}\|p)$ | $0.68\log(0.34/0.49)+0.32\log16$ | $0.638716$ |
| $\operatorname{KL}(q_{left}\|p)$ | $0.96\log(0.96/0.49)+0.02\log(0.02/0.49)$ | $0.581653$ |

Forward KL prefers cover; reverse KL prefers left among these candidates.
Reversing KL is not negating a forward score: it also changes the weights.
For fixed discrete $p$, add and subtract $\log p$:

$$
H(p,q)=-\sum_xp(x)\log q(x)
=-\sum_xp(x)\log p(x)+\sum_xp(x)\log\frac{p(x)}{q(x)}
=H(p)+\operatorname{KL}(p\|q).
$$

Only $H(p)$ stays constant when optimizing $q$.
Categorical log-loss minimization and forward-KL minimization therefore have the same optimizer over the chosen model family.

### Remove candidate support

**New task:** replace the left candidate with $(0.98,0.02,0)$.
**Picture and change:** the target's final outcome still has mass $0.49$, but the candidate now assigns zero.
**Check:** forward KL becomes infinite; reverse KL remains $0.98\log2\approx0.679284$ nats.
**Use when:** truncation or exact zeros remove support; exchanging arguments changes the requirement.

### Expand the candidate set

**New task:** permit $q=p$ as another candidate.
Every log ratio is now zero for that candidate.
**Check:** both objectives attain zero at $p$, beating the original choices.
**Use when:** a richer approximation family becomes available; reverse KL does not inevitably select one mode.

### Failure and transfer

For $p=(0.9,0.1)$ and $q=(0.5,0.5)$, the wrong average $\mathbb E_q[\log(p/q)]$ is $-0.510826$.
The actual $\operatorname{KL}(p\|q)$ is $0.368064$.
Wrong sampling weights changed the quantity, not the nonnegativity theorem.

**Reconstruct:** fixed $p$ has entropy $0.7$ nats and cross-entropy $1.2$ nats. What is KL?

<details><summary>Answer</summary>

It is $0.5$ nats.
Minimizing cross-entropy over $q$ also minimizes forward KL only because this $p$, and hence its entropy, stays fixed.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

### Why the divergence is nonnegative

For positive $r$, $-\log r\ge1-r$.
Apply this with $r=q(x)/p(x)$ wherever $p(x)>0$:

$$
\operatorname{KL}(p\|q)\ge
\sum_{p(x)>0}\bigl(p(x)-q(x)\bigr)
=1-\sum_{p(x)>0}q(x)\ge0.
$$

Equality requires $q(x)=p(x)$ on the positive support and no remaining candidate mass outside it.
Zero candidate probability at positive target mass instead gives infinite KL immediately.
The coding interpretation compares expected ideal code lengths under the same sampling law; base-two logarithms give bits rather than nats.

### Continuous laws and empirical samples

For general measures, finite KL requires absolute continuity $p\ll q$ and suitable integrability.
For densities, use $\int p\log(p/q)$ against the common base measure.
An atomic empirical measure has infinite KL to a continuous law assigning zero probability to every observed point.
Empirical negative log density remains meaningful; it is not that literal finite empirical-to-continuous KL identity.

## Common usage in ML

| Use case | Direction |
|---------|-----------|
| Classification cross-entropy loss | Forward $\mathrm{KL}(\text{data} \| \text{model})$ |
| Variational inference | Approximate posterior $q(z\mid x)$ versus model posterior $p(z\mid x)$ |
| Policy regularization | Name both policies, expectation, and penalty direction explicitly |
| Knowledge distillation | Teacher versus student distributions at the specified temperature |
| t-SNE | High-dimensional versus low-dimensional pairwise probability distributions |

PPO's clipped surrogate is not generally an exact KL constraint.
Information bottleneck objectives also use distinct joint, marginal, and conditional laws; name them before applying an identity.

## Common pitfalls

Jensen-Shannon divergence averages $\operatorname{KL}(p\|m)$ and $\operatorname{KL}(q\|m)$ with $m=(p+q)/2$.
That symmetric bounded quantity is not KL with swapped arguments or a universal GAN objective.
Finite-sample density estimates introduce their own support choices and estimation bias.
Changing the feasible family can matter as much as divergence direction.

*Related: [entropy, mutual information, and information gain](/concepts/entropy-mutual-information/), [cross-entropy and softmax](/concepts/cross-entropy-softmax/), and [variational autoencoders](/concepts/variational-autoencoders/).*

</div>
