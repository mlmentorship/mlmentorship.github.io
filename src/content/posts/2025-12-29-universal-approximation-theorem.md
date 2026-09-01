---
title: "Universal approximation theorem"
description: "A neural network with one hidden layer and enough units can approximate any continuous function on a bounded domain. What it does and doesn't say about deep learning."
date: "2025-12-29"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

The **universal approximation theorem** ([Cybenko, 1989](https://link.springer.com/article/10.1007/BF02551274); [Hornik, 1991](https://www.sciencedirect.com/science/article/abs/pii/089360809190009T)) applies to continuous functions on a compact subset of $\mathbb{R}^n$. A feed-forward network with one sufficiently wide hidden layer and a suitable non-polynomial activation can approximate any such function to arbitrary accuracy.

UAT is often quoted as "neural networks can learn anything." That is a misleading summary; the theorem is an existence result, not a guarantee that:

- training will find the approximating network,
- the network has reasonable size,
- it generalizes from finite samples,
- it is practical for the input dimension you care about.

Knowing what UAT does and doesn't promise is a senior-level expectation; the wrong reading shows up regularly in interviews.

## What the theorem says (precisely)

For any continuous function $f: K \to \mathbb{R}$ on a compact $K \subset \mathbb{R}^n$ and any $\varepsilon > 0$, there exists a network

$$
g(x) = \sum_{i=1}^{N} c_i \sigma(w_i^\top x + b_i)
$$

with finite width $N$ such that $\sup_{x \in K} |g(x) - f(x)| < \varepsilon$, where $\sigma$ is a continuous sigmoidal activation in Cybenko's formulation. Later results broaden the sufficient activation conditions.

**Learning objective:** read universal approximation as a uniform error guarantee for some finite network on a compact domain, then separate that existence claim from width, optimization, generalization, and extrapolation.

<!-- visual:uat-existence-error-band -->
<figure class="learning-figure plot-panel" aria-labelledby="uat-error-band-title">
	<p class="visual-kicker">The quantifiers are the lesson</p>
	<p class="visual-title" id="uat-error-band-title">What does “there exists an approximating network” actually guarantee?</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="uat-error-band-svg-title uat-error-band-svg-desc">
		<title id="uat-error-band-svg-title">A network approximation stays inside a uniform error band only on a compact domain</title>
		<desc id="uat-error-band-svg-desc">On the compact interval K from a to b, a solid piecewise-linear target function is surrounded by a shaded tolerance band extending twelve drawing units above and below it. A dashed piecewise-linear network approximation stays inside the band. Its vertical error at every point is at most six units, so the maximum error is below epsilon. Below the plot, one box states what the theorem provides: some finite width and weights exist. A second box states what it does not provide: a small width, weights found by SGD, generalization from samples, or behavior outside K.</desc>
		<rect class="viz-plot-bg" x="8" y="30" width="344" height="246" rx="5"></rect>
		<text class="viz-axis-label" x="14" y="18">ON K, THE APPROXIMATION MUST STAY INSIDE THE ε BAND EVERYWHERE</text>
		<path d="M35 118L70 63L105 43L140 118L175 173L210 193L245 118L280 63L325 118L325 142L280 87L245 142L210 217L175 197L140 142L105 67L70 87L35 142Z" style="fill:var(--viz-focus-bg);stroke:none"></path>
		<path d="M35 130L70 75L105 55L140 130L175 185L210 205L245 130L280 75L325 130" class="viz-roc-curve"></path>
		<path d="M35 126L70 81L105 61L140 124L175 179L210 199L245 136L280 69L325 124" class="viz-pr-curve" stroke-dasharray="7 5"></path>
		<path d="M35 38V250M325 38V250" class="viz-baseline"></path>
		<path d="M27 250H333" class="viz-axis"></path>
		<text class="viz-callout" x="40" y="109">target f: solid</text>
		<text class="viz-callout" x="214" y="188">network g: dashed</text>
		<text class="viz-label" x="176" y="47">upper tolerance f(x) + ε</text>
		<text class="viz-label" x="177" y="221">lower tolerance f(x) − ε</text>
		<path d="M105 55H119M105 61H119M116 55V61" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
		<text class="viz-callout" x="123" y="62">|g − f| ≤ 6 &lt; ε = 12</text>
		<text class="viz-callout" x="31" y="268">a</text>
		<text class="viz-callout" x="321" y="268">b</text>
		<text class="viz-axis-label" x="180" y="268" text-anchor="middle">COMPACT DOMAIN K = [a, b]</text>
		<rect class="viz-node viz-node--output" x="12" y="296" width="336" height="48" rx="4"></rect>
		<text class="viz-callout" x="24" y="316">THE THEOREM PROVIDES EXISTENCE</text>
		<text class="viz-label" x="24" y="334">Some finite width N and some weights make supₓ∈K |g(x) − f(x)| &lt; ε.</text>
		<rect class="viz-node viz-node--warning" x="12" y="356" width="336" height="62" rx="4"></rect>
		<text class="viz-callout" x="24" y="376">THE THEOREM DOES NOT PROVIDE A PRACTICAL RECIPE</text>
		<text class="viz-label" x="24" y="395">No small N · no SGD guarantee · no finite-sample generalization</text>
		<text class="viz-label" x="24" y="410">No constraint outside K</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> choose the compact domain and tolerance first. The theorem says that some finite network can keep its largest vertical error below that tolerance everywhere in the domain. It does not tell you how wide the network is, how to find its weights, whether it generalizes from samples, or what happens outside the domain. The target, approximation, and coordinates are an original construction checked against <a href="https://hal.science/hal-03753170v1">Cybenko (1989)</a>.</figcaption>
</figure>

Modern extensions:

- **ReLU networks** are also universal approximators ([Pinkus, 1999](https://www.cambridge.org/core/journals/acta-numerica/article/abs/approximation-theory-of-the-mlp-model-in-neural-networks/18072C558C8410C4F92A82BCC8FC8CF9); [Lu et al., 2017](https://arxiv.org/abs/1709.02540)).
- **Deep networks** with bounded width can be universal [(Lu et al., 2017)](https://arxiv.org/abs/1709.02540): width $\ge n + 4$ suffices for some classes.

## What the theorem does **not** say

1. **Width may be exponential.** UAT does not bound $N$. For some functions, the required width is exponential in input dimension.
2. **Training is not guaranteed.** UAT is non-constructive. It proves existence, not how SGD finds it.
3. **Generalization is not addressed.** A perfect fit on training data is not the same as predicting on test data.
4. **Deep beats wide for some functions.** UAT applies to wide-shallow nets; depth gives exponential efficiency for many natural functions [(Telgarsky, 2016)](https://arxiv.org/abs/1602.04485).

## Why deep nets are practically necessary

If shallow nets are universal, why use deep ones? Two reasons:

- **Compositional efficiency**: many functions of practical interest (image features, language structure) are naturally compositional. Deep nets express them with polynomially fewer units than shallow nets ([Mhaskar & Poggio, 2016](https://arxiv.org/abs/1603.00988); [Eldan & Shamir, 2016](https://arxiv.org/abs/1512.03965)).
- **Optimization landscape**: SGD finds good solutions in deep over-parameterized networks more reliably than in narrow shallow ones. Empirically and per modern theory (NTK, lottery ticket, etc.).

So UAT justifies "neural networks can fit anything in principle." Practical deep learning relies on additional, separately-justified properties.

## Related theoretical results

- **Barron's theorem** (1993): for functions with bounded "Barron norm," the approximation error of a width-$N$ shallow net is $O(1/\sqrt{N})$. Independent of input dimension. Constructive guarantee for a restricted function class.
- **Kolmogorov–Arnold theorem** (1957): continuous functions on $[0,1]^n$ can be exactly represented as a sum of compositions of single-variable continuous functions. Inspired KAN architectures (2024).
- **Width-bounded ReLU UAT**: width $n + 4$ is sufficient for universality [(Lu et al., 2017)](https://arxiv.org/abs/1709.02540).

## What to say in interviews

If asked "do neural networks really learn anything?":

1. State UAT precisely (one hidden layer, non-polynomial activation, compact domain).
2. Note that it is non-constructive and bounds nothing about width or trainability.
3. Argue that practical deep learning relies on (a) compositional efficiency of depth, (b) the optimization landscape of over-parameterized networks, and (c) inductive biases of architectures (CNNs for translation invariance, transformers for sequences).

That sequence demonstrates senior-level understanding rather than sloganeering.

## Common pitfalls

- **Citing UAT as a guarantee that any NN learns its task.** UAT says some network *exists*; SGD may not find it.
- **Using UAT to justify wide-shallow nets.** Empirically, depth helps; UAT alone doesn't predict that.
- **Ignoring the compactness assumption.** UAT is for compact domains; behavior outside the training support is unconstrained.
