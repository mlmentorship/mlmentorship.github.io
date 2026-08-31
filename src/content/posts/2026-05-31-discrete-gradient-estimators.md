---
title: "Discrete gradient estimators"
description: "How to get gradients through a sampling step over discrete variables, where the reparameterization trick doesn't apply. Covers the score-function (REINFORCE) estimator, the straight-through estimator, and Gumbel-Softmax."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Discrete gradient estimators approximate $\nabla_\theta \mathbb{E}_{z \sim p_\theta}[f(z)]$ when $z$ is **discrete**, the case where you cannot reparameterize the sample as a smooth function of $\theta$ and noise. The three you must know: **REINFORCE** (score function), **Gumbel-Softmax** (continuous relaxation), and the **straight-through estimator**.

The [reparameterization trick](/questions/reparameterization-trick/) handles continuous latents (Gaussian VAEs). But many models sample **discrete** objects: categorical latents, hard attention, tokens, architecture choices, RL actions. You can't push a gradient through `argmax` or a categorical sample, so you need an estimator. This is the deep-DL follow-up to "explain the reparameterization trick," and it underpins RLHF (which uses the score-function estimator) and discrete latent-variable models.

## The core problem

We want $\nabla_\theta \mathbb{E}_{z \sim p_\theta(z)}[f(z)]$. The expectation is a sum over discrete $z$; the sampling operation is non-differentiable. The two families of solutions trade **bias** for **variance**.

<!-- visual:discrete-gradients-forward-backward-paths -->
<figure class="learning-figure" aria-labelledby="discrete-gradient-paths-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="discrete-gradient-paths-title">Where does each estimator send the backward signal?</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 570" role="img" aria-labelledby="discrete-gradient-svg-title discrete-gradient-svg-desc">
			<title id="discrete-gradient-svg-title">Forward and backward paths for three discrete gradient estimators</title>
			<desc id="discrete-gradient-svg-desc">Three stacked computation paths compare REINFORCE, Gumbel-Softmax, and straight-through estimation. Solid arrows point right and show the forward computation. Dashed arrows point left and show the backward signal. REINFORCE evaluates a hard categorical sample and sends the sampled reward directly to the log probability without differentiating through the sample or objective. Gumbel-Softmax replaces the hard sample with a soft temperature-controlled sample and differentiates through the entire relaxed path. Straight-through uses a hard sample in the forward path but routes the backward signal through a separate soft surrogate.</desc>
			<defs>
				<marker id="discrete-forward-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
				<marker id="discrete-backward-arrow" markerWidth="7" markerHeight="7" refX="1" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M7 0L0 3.5L7 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="5" y="5" width="350" height="170" rx="6"></rect>
			<text class="viz-axis-label" x="18" y="27">1 · REINFORCE: HARD FORWARD, SCORE BACKWARD</text>
			<rect class="viz-node viz-node--input" x="18" y="54" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="60" y="75">π<tspan baseline-shift="sub" font-size="9">θ</tspan></text><text class="viz-node-value" x="60" y="92">probabilities</text>
			<rect class="viz-node viz-node--focus" x="137" y="54" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="179" y="75">hard z</text><text class="viz-node-value" x="179" y="92">sample category</text>
			<rect class="viz-node viz-node--output" x="256" y="54" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="298" y="75">f(z)</text><text class="viz-node-value" x="298" y="92">sampled value</text>
			<path d="M102 78H133" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#discrete-forward-arrow)"></path>
			<path d="M221 78H252" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#discrete-forward-arrow)"></path>
			<path d="M298 108C298 153 60 153 60 108" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#discrete-backward-arrow)"></path>
			<text class="viz-gradient-label" x="179" y="137">f(z) · ∇θ log pθ(z)</text>
			<text class="viz-edge-label" x="179" y="160">bypasses derivatives of hard z and f</text>
			<rect class="viz-plot-bg" x="5" y="191" width="350" height="170" rx="6"></rect>
			<text class="viz-axis-label" x="18" y="213">2 · GUMBEL-SOFTMAX: SOFT FORWARD + BACKWARD</text>
			<rect class="viz-node viz-node--input" x="18" y="240" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="60" y="261">log π + g</text><text class="viz-node-value" x="60" y="278">fixed noise g</text>
			<rect class="viz-node viz-node--focus" x="137" y="240" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="179" y="261">soft y<tspan baseline-shift="sub" font-size="9">τ</tspan></text><text class="viz-node-value" x="179" y="278">relaxed sample</text>
			<rect class="viz-node viz-node--output" x="256" y="240" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="298" y="261">f(y<tspan baseline-shift="sub" font-size="9">τ</tspan>)</text><text class="viz-node-value" x="298" y="278">must be smooth</text>
			<path d="M102 264H133" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#discrete-forward-arrow)"></path>
			<path d="M221 264H252" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#discrete-forward-arrow)"></path>
			<path d="M298 294V320H60V294" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#discrete-backward-arrow)"></path>
			<text class="viz-gradient-label" x="179" y="314">pathwise gradient through the relaxation</text>
			<text class="viz-edge-label" x="179" y="345">low variance · biased for the discrete objective</text>
			<rect class="viz-plot-bg" x="5" y="377" width="350" height="188" rx="6"></rect>
			<text class="viz-axis-label" x="18" y="399">3 · STRAIGHT-THROUGH: HARD FORWARD, SOFT BACKWARD</text>
			<rect class="viz-node viz-node--input" x="18" y="426" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="60" y="447">logits</text><text class="viz-node-value" x="60" y="464">parameters θ</text>
			<rect class="viz-node viz-node--focus" x="137" y="426" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="179" y="447">hard z</text><text class="viz-node-value" x="179" y="464">forward only</text>
			<rect class="viz-node viz-node--output" x="256" y="426" width="84" height="48" rx="5"></rect>
			<text class="viz-node-label" x="298" y="447">f(z)</text><text class="viz-node-value" x="298" y="464">sees hard value</text>
			<path d="M102 450H133" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#discrete-forward-arrow)"></path>
			<path d="M221 450H252" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#discrete-forward-arrow)"></path>
			<rect class="viz-node" x="137" y="506" width="84" height="38" rx="5" style="stroke-dasharray:5 3"></rect>
			<text class="viz-node-value" x="179" y="522">soft surrogate y<tspan baseline-shift="sub" font-size="8">τ</tspan></text><text class="viz-node-value" x="179" y="536">backward only</text>
			<path d="M298 480V525H225" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#discrete-backward-arrow)"></path>
			<path d="M133 525H60V480" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#discrete-backward-arrow)"></path>
			<text class="viz-edge-label" x="277" y="548">deliberate mismatch → biased</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> solid arrows are what the model computes; dashed arrows are the training signal. REINFORCE jumps around the non-differentiable sample using its log probability, Gumbel-Softmax makes the whole path soft, and straight-through keeps the hard forward value while borrowing a soft backward path. The construction is original; definitions were checked against Williams (1992), Jang et al. (2017), Maddison et al. (2017), and Bengio et al. (2013).</figcaption>
</figure>

## 1. Score-function estimator (REINFORCE / likelihood ratio)

Use the log-derivative identity $\nabla_\theta p_\theta(z) = p_\theta(z)\, \nabla_\theta \log p_\theta(z)$:

$$
\nabla_\theta \mathbb{E}_{z}[f(z)] = \mathbb{E}_{z \sim p_\theta}\big[\, f(z)\, \nabla_\theta \log p_\theta(z)\,\big].
$$

**Unbiased**, requires only that you can sample $z$ and evaluate $\log p_\theta(z)$; $f$ can be a black box (non-differentiable, even an environment reward).

The catch is **high variance**. Mitigations:

- **Baselines / control variates**: subtract a baseline $b$ that doesn't depend on $z$: $(f(z) - b)\nabla_\theta \log p_\theta(z)$. Still unbiased (since $\mathbb{E}[\nabla_\theta \log p_\theta] = 0$), lower variance. The value-function baseline in actor-critic is exactly this.
- More samples, advantage normalization, etc.

This estimator **is** policy-gradient RL. REINFORCE, A2C, and PPO are all score-function estimators with progressively better variance control.

## 2. Gumbel-Softmax (Concrete distribution)

Relax the discrete sample into a continuous one you *can* reparameterize. The **Gumbel-Max trick** says a categorical sample equals

$$
z = \operatorname*{arg\,max}_i \big(\log \pi_i + g_i\big), \qquad g_i \sim \text{Gumbel}(0,1).
$$

Replace the non-differentiable `argmax` with a temperature-$\tau$ **softmax**:

$$
y_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}.
$$

Now $y$ is a differentiable, reparameterized sample (a point on the simplex). As $\tau \to 0$, $y$ approaches a one-hot vector but the gradient variance blows up; as $\tau$ grows, samples are smooth but biased toward uniform. You **anneal** $\tau$ downward during training. **Low variance, biased.**

## 3. Straight-through estimator (STE)

Forward pass: use the **hard** discrete value (e.g. `argmax`, or a threshold). Backward pass: pretend the operation was the identity (or the softmax), and pass the gradient straight through.

$$
\text{forward: } z = \text{one\_hot}(\arg\max), \qquad \text{backward: } \frac{\partial z}{\partial \text{logits}} \approx \frac{\partial \,\text{softmax}}{\partial \text{logits}}.
$$

**Straight-Through Gumbel-Softmax** combines both: hard one-hot forward, soft Gumbel-Softmax gradient backward, so the rest of the network sees a genuine discrete sample. STE is **biased** (the backward op isn't the true derivative) but cheap and empirically effective; it is the workhorse behind **VQ-VAE** codebook training and binarized/quantized networks.

## The bias-variance tradeoff

| Estimator | Bias | Variance | Needs differentiable $f$? | Typical use |
| --- | --- | --- | --- | --- |
| **Score function (REINFORCE)** | Unbiased | High | No | RL, RLHF, black-box reward |
| **Gumbel-Softmax** | Biased ($\tau>0$) | Low | Yes | Discrete latents (categorical VAE) |
| **Straight-through** | Biased | Low | Yes (via surrogate) | VQ-VAE, quantization, hard attention |

The dividing question: **can you differentiate $f$?** If not (an environment, a metric, a sampled-then-scored pipeline), you're forced onto the score-function estimator. If you can, the relaxation methods give far lower variance.

## What an interviewer expects you to say

1. State *why* reparameterization fails for discrete $z$ (you can't write a discrete sample as a smooth function of noise and $\theta$).
2. Give the **score-function estimator** with the $\nabla \log p$ identity, that it's **unbiased but high-variance**, and that **baselines** reduce variance without adding bias.
3. Explain **Gumbel-Softmax** as the reparameterizable relaxation with a temperature you anneal (**biased, low variance**).
4. Describe the **straight-through estimator** (hard forward, soft/identity backward) and that it trains **VQ-VAE** and quantized nets.
5. Connect to practice: **RLHF uses score-function (PPO)** because text is discrete and a 50K-way Gumbel-Softmax is impractical; **DPO** sidesteps sampling entirely.

## Common confusions

- **"You can just backprop through argmax."** Its gradient is zero almost everywhere; that's the whole problem.
- **"REINFORCE is biased."** It's unbiased; its issue is variance. Baselines fix variance, not bias.
- **"Gumbel-Softmax is exact."** It's biased for any $\tau > 0$; only the $\tau \to 0$ limit is exact, and there the gradient is uselessly high-variance.
- **"Straight-through has a principled gradient."** It doesn't; it's a useful heuristic (the backward op deliberately mismatches the forward op).
- **"These are RL-only / VAE-only tricks."** They're general: hard attention, neural architecture search, discrete communication, and quantization all use them.

---

*Related: [Explain the reparameterization trick](/questions/reparameterization-trick/), [Policy gradient](/concepts/policy-gradient/), [PPO](/concepts/ppo/), [Variational autoencoders](/concepts/variational-autoencoders/), [Quantization](/concepts/quantization/).*
