---
title: "Weight initialization (Kaiming, Xavier)"
description: "Set the initial variance of each layer's weights so that activations and gradients neither explode nor vanish through depth. The single most impactful one-line decision in deep nets."
date: "2026-05-05"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Initialize each layer's weights from a distribution whose variance is set so the variance of activations (forward pass) and gradients (backward pass) stays approximately constant from layer to layer. Two standard schemes: **Xavier/Glorot** for tanh/sigmoid layers, **Kaiming/He** for ReLU-family layers.

If weights are too small, activations shrink toward zero through depth and gradients vanish. If too large, activations explode and gradients blow up. With either failure, training stalls or diverges in the first few hundred steps.

A correct init lets a 24-layer transformer train to convergence with vanilla SGD or Adam; an incorrect init makes the same architecture untrainable without ad-hoc fixes (warmup hacks, smaller LR, etc.).

## The variance argument

For a linear layer $y = W x$ with $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, $x$ zero-mean with variance $\sigma_x^2$, and $W$ drawn iid with mean 0 and variance $\sigma_W^2$:

$$
\text{Var}(y_i) = d_\text{in} \cdot \sigma_W^2 \cdot \sigma_x^2.
$$

To preserve variance ($\sigma_y^2 = \sigma_x^2$), pick $\sigma_W^2 = 1 / d_\text{in}$.

The same argument on the backward pass gives $\sigma_W^2 = 1 / d_\text{out}$. Compromise:

$$
\sigma_W^2 = \frac{2}{d_\text{in} + d_\text{out}} \quad \text{(Xavier/[Glorot, 2010](https://proceedings.mlr.press/v9/glorot10a.html))}
$$

For ReLU activations, half the activations are zeroed out, halving variance. Compensate with a factor of 2:

$$
\sigma_W^2 = \frac{2}{d_\text{in}} \quad \text{([Kaiming/He, 2015](https://arxiv.org/abs/1502.01852), "fan-in" mode)}
$$

## Practical defaults

| Layer type | Init |
|------------|------|
| Linear, ReLU/GELU activation | Kaiming-normal, fan-in |
| Linear, tanh/sigmoid | Xavier-uniform |
| Conv, ReLU | Kaiming-normal, fan-in |
| Embeddings | $\mathcal{N}(0, 0.02^2)$ for transformers; $\mathcal{N}(0, 1)$ when followed by LayerNorm |
| LayerNorm $\gamma$ | 1 |
| LayerNorm $\beta$ | 0 |
| Bias | 0 |

Most modern frameworks default to Kaiming-uniform for `nn.Linear` (PyTorch). For transformers, GPT-style models often add a per-residual scaling $1/\sqrt{2 \cdot N_\text{layers}}$ on the output projections to keep residual-stream variance bounded with depth.

## Special cases

- **Residual connections**: with N layers, the residual stream's variance grows linearly with depth unless the contributions from each block are downscaled. GPT-2 / GPT-3 scale output projections by $1/\sqrt{N}$.
- **Identity init for recurrent** [(Le et al., 2015)](https://arxiv.org/abs/1504.00941): initialize the recurrent weight matrix to the identity to make RNNs behave like feed-forward at $t=0$.
- **Orthogonal init**: weight matrices initialized to orthogonal matrices preserve norms exactly. Used in some RL policy networks.

## Common pitfalls

- **Using PyTorch's default `nn.Linear` for a transformer without checking it.** The default is Kaiming-uniform with the wrong fan; many transformer codebases override it with $\mathcal{N}(0, 0.02^2)$.
- **Initializing bias to nonzero.** Almost never helps; can break symmetry breaking arguments.
- **Forgetting to scale residual outputs.** Without it, deep transformers produce huge residual-stream values at init.
- **Trusting "it trains" as proof of correct init.** It might converge slower than a properly initialized run.
