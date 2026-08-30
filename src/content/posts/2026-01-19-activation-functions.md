---
title: "Activation functions"
description: "ReLU, GELU, swish, sigmoid, tanh. What each does, why GELU/swish replaced ReLU in transformers, and when to use which."
date: "2026-01-19"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An **activation function** is a (usually) elementwise nonlinearity applied between linear layers in a neural network. Without it, stacking linear layers collapses to a single linear layer (no expressive power gain). The choice of activation shapes optimization, gradient flow, and final accuracy.

## The standard family

| Activation | Formula | Range | Use today |
|-----------|---------|-------|-----------|
| **Sigmoid** | $\sigma(z) = 1/(1 + e^{-z})$ | $(0, 1)$ | Output of binary classifier; gates in LSTMs/GRUs. Hidden layers: avoid (saturating gradients). |
| **Tanh** | $\tanh(z)$ | $(-1, 1)$ | RNN hidden (legacy); zero-centered version of sigmoid. |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | Default for CNNs and MLPs; cheap, fast. |
| **Leaky ReLU** | $\max(\alpha z, z)$, $\alpha \approx 0.01$ | $(-\infty, \infty)$ | Avoids "dying ReLU" by leaking negative values. |
| **ELU** | $z$ for $z > 0$, $\alpha (e^z - 1)$ for $z \le 0$ | $(-\alpha, \infty)$ | Smooth and zero-centered. Slightly slower than ReLU. |
| **GELU** | $z \cdot \Phi(z)$ where $\Phi$ is standard normal CDF | $\mathbb{R}$ | Default in transformers (BERT, GPT-1/2/3). |
| **Swish / SiLU** | $z \cdot \sigma(z)$ | $\mathbb{R}$ | Default in modern decoder LLMs (Llama, Mistral). |
| **Softmax** | $\exp(z_i) / \sum_j \exp(z_j)$ | simplex | Output of multi-class classifier; not used in hidden layers. |

## Why ReLU won (then why GELU/swish replaced it)

**ReLU** [(Nair & Hinton, 2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) helped mitigate vanishing gradients in deep networks:

- Gradient is exactly 1 in the active region (no saturation).
- Computationally trivial: a single $\max$.
- Sparse activations (~half are zero): biological intuition + computational efficiency.

But ReLU has the **dying ReLU problem**: a neuron stuck at $z < 0$ has gradient 0 forever and never recovers.

**GELU** [(Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415) and **swish / SiLU** [(Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941) are smooth alternatives to ReLU. They retain small negative outputs rather than setting the entire negative half-line to zero. Their derivatives can still equal zero at isolated points, but they do not have ReLU's half-line with an identically zero derivative. Both are widely used in transformer feed-forward blocks.

<!-- visual:activation-gradient-regions -->
<figure class="learning-figure" aria-labelledby="activation-regions-title">
	<p class="visual-kicker">Optimization intuition</p>
	<p class="visual-title" id="activation-regions-title">Output shape determines where backpropagated gradients shrink or stop.</p>
	<div class="visual-grid--two">
		<section class="visual-panel plot-panel" aria-labelledby="sigmoid-panel-heading">
			<h4 id="sigmoid-panel-heading">Sigmoid: two saturating tails</h4>
			<p>The slope approaches zero at both output limits.</p>
			<svg viewBox="0 0 320 230" role="img" aria-labelledby="sigmoid-svg-title sigmoid-svg-desc">
				<title id="sigmoid-svg-title">Logistic sigmoid from negative four to positive four</title>
				<desc id="sigmoid-svg-desc">The sigmoid rises smoothly through one half at input zero and flattens toward output zero on the left and output one on the right. Both flat tails are labeled as saturation regions where the derivative approaches zero.</desc>
				<rect class="viz-plot-bg" x="42" y="24" width="250" height="166" rx="3"></rect>
				<path class="viz-gridline" d="M42 123H292 M42 57H292"></path>
				<path class="viz-axis" d="M167 24V190 M42 190H292"></path>
				<path class="viz-roc-curve" d="M42 187.6 L49.8 186.9 L57.6 186.1 L65.4 185 L73.3 183.7 L81.1 182 L88.9 179.9 L96.7 177.3 L104.5 174.1 L112.3 170.3 L120.1 165.7 L127.9 160.3 L135.8 154.1 L143.6 147.2 L151.4 139.7 L159.2 131.6 L167 123.3 L174.8 115 L182.6 107 L190.4 99.4 L198.3 92.5 L206.1 86.4 L213.9 81 L221.7 76.4 L229.5 72.6 L237.3 69.4 L245.1 66.8 L252.9 64.7 L260.8 63 L268.6 61.6 L276.4 60.6 L284.2 59.7 L292 59.1"></path>
				<text class="viz-callout" x="48" y="130">left tail:</text>
				<text class="viz-callout" x="48" y="144">σ′(x) → 0</text>
				<text class="viz-callout" x="185" y="46">saturated: σ′(x) → 0</text>
				<circle class="viz-operating-point" cx="167" cy="123" r="4"></circle>
				<text class="viz-label" x="173" y="119">σ(0) = 0.5</text>
				<text class="viz-label" x="38" y="207">−4</text><text class="viz-label" x="163" y="207">0</text><text class="viz-label" x="288" y="207">4</text>
				<text class="viz-label" x="25" y="193">0</text><text class="viz-label" x="17" y="127">0.5</text><text class="viz-label" x="25" y="61">1</text>
				<text class="viz-axis-label" x="281" y="222">input x</text>
				<text class="viz-axis-label" x="46" y="20">output</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel" aria-labelledby="rectifier-panel-heading">
			<h4 id="rectifier-panel-heading">ReLU vs exact GELU</h4>
			<p>A half-line of zero slope differs from one smooth shoulder.</p>
			<svg viewBox="0 0 320 230" role="img" aria-labelledby="rectifier-svg-title rectifier-svg-desc">
				<title id="rectifier-svg-title">ReLU and exact GELU from negative three to positive three</title>
				<desc id="rectifier-svg-desc">ReLU is exactly zero with zero derivative for every negative input, then follows a straight line after zero. Exact GELU, x times the standard normal cumulative distribution, has small negative outputs, a shallow minimum near negative three quarters, passes through zero, and approaches the same positive linear trend. ReLU is solid and GELU is dashed.</desc>
				<rect class="viz-plot-bg" x="42" y="24" width="250" height="166" rx="3"></rect>
				<path class="viz-gridline" d="M42 125H292 M42 82H292 M42 39H292"></path>
				<path class="viz-axis" d="M167 24V190 M42 168.4H292"></path>
				<path class="viz-roc-curve" d="M42 168.4 L83.7 168.4 L125.3 168.4 L167 168.4 L208.7 125.1 L250.3 81.9 L292 38.6"></path>
				<path class="viz-pr-curve" stroke-dasharray="6 4" d="M42 168.6 L47.2 168.6 L52.4 168.7 L57.6 168.9 L62.8 169 L68 169.3 L73.3 169.6 L78.5 169.9 L83.7 170.3 L88.9 170.8 L94.1 171.4 L99.3 172 L104.5 172.7 L109.7 173.4 L114.9 174.1 L120.1 174.7 L125.3 175.2 L130.5 175.6 L135.8 175.7 L141 175.6 L146.2 175 L151.4 174.1 L156.6 172.7 L161.8 170.8 L167 168.4 L172.2 165.4 L177.4 161.9 L182.6 157.9 L187.8 153.4 L193 148.5 L198.3 143.3 L203.5 137.8 L208.7 132 L213.9 126.1 L219.1 120 L224.3 113.9 L229.5 107.8 L234.7 101.8 L239.9 95.7 L245.1 89.8 L250.3 83.9 L255.5 78 L260.8 72.3 L266 66.6 L271.2 60.9 L276.4 55.4 L281.6 49.8 L286.8 44.3 L292 38.8"></path>
				<text class="viz-callout" x="49" y="155">ReLU: dead region, f′(x) = 0</text>
				<text class="viz-callout" x="203" y="112">ReLU: solid</text>
				<text class="viz-callout" x="180" y="137">GELU: dashed</text>
				<text class="viz-label" x="75" y="143">smooth negative shoulder</text>
				<circle class="viz-operating-point" cx="167" cy="168.4" r="4"></circle>
				<text class="viz-label" x="172" y="181">both pass through (0, 0)</text>
				<text class="viz-label" x="38" y="207">−3</text><text class="viz-label" x="163" y="207">0</text><text class="viz-label" x="288" y="207">3</text>
				<text class="viz-label" x="25" y="172">0</text><text class="viz-label" x="25" y="129">1</text><text class="viz-label" x="25" y="86">2</text><text class="viz-label" x="25" y="43">3</text>
				<text class="viz-axis-label" x="281" y="222">input x</text>
				<text class="viz-axis-label" x="46" y="20">output</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> sigmoid loses slope in both tails, while ReLU preserves slope for positive inputs but kills it for every negative input. Exact GELU replaces that dead half-line with a smooth, slightly negative shoulder, not with an everywhere-nonzero derivative.</figcaption>
</figure>

In 2026:

- **CNNs / MLPs**: still mostly ReLU.
- **Transformers**: GELU (BERT-era) or SwiGLU (modern Llama-style decoders).
- **RNNs**: tanh / sigmoid for gates (legacy); RNNs are largely deprecated for new work.

## SwiGLU and gated activations

Modern decoder LLMs (Llama 1/2/3, Mistral, Qwen) use **SwiGLU** in the FFN:

$$
\text{FFN}(x) = (\text{swish}(W_1 x) \odot (W_2 x)) W_3.
$$

Two parallel linear projections, one passed through swish, then elementwise product, then a third linear projection. Slightly more parameters per FFN block than the original $W_1, W_2$ design, but better training dynamics. GLU = "gated linear unit." Now standard.

## When to use which output activation

| Task | Output activation |
|------|------------------|
| Binary classification | Sigmoid |
| Multi-class classification | Softmax |
| Multi-label classification | Sigmoid (independent binary heads) |
| Regression (unbounded) | Identity (no activation) |
| Regression (bounded $[0, 1]$) | Sigmoid |
| Probability over a discrete distribution | Softmax |
| Embedding output | Identity, then L2-normalize |

## Common pitfalls

- **Putting ReLU on the output.** A regression with non-negative range should use ReLU or softplus on output; for general regression, no activation.
- **Sigmoid in hidden layers of deep nets.** Saturates → vanishing gradients → no learning past a few layers.
- **Picking exotic activations to chase 0.5% accuracy.** The activation choice rarely matters compared to data, regularization, and architecture.
- **Forgetting GELU has exact and approximate forms.** The CDF-based and tanh-based formulas are numerically close but not identical; check your framework when exact reproducibility matters.

## Related

- [Weight initialization](/concepts/weight-initialization/). Initialization is activation-dependent.
- [Exploding and vanishing gradients](/concepts/exploding-vanishing-gradients/). What motivates ReLU and gated activations.
