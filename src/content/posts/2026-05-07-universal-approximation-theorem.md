---
title: "Universal approximation theorem"
description: "A neural network with one hidden layer and enough units can approximate any continuous function on a bounded domain. What it does and doesn't say about deep learning."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

The **universal approximation theorem** ([Cybenko, 1989](https://link.springer.com/article/10.1007/BF02551274); [Hornik, 1991](https://www.sciencedirect.com/science/article/abs/pii/089360809190009T)) states that a feed-forward neural network with a single hidden layer of sigmoidal (or other non-polynomial) units can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy, provided the layer has enough units.

## Why it matters

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

with finite width $N$ such that $\sup_{x \in K} |g(x) - f(x)| < \varepsilon$, where $\sigma$ is any non-polynomial bounded activation.

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
