---
title: "Activation functions"
description: "ReLU, GELU, swish, sigmoid, tanh. What each does, why GELU/swish replaced ReLU in transformers, and when to use which."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

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

**ReLU** [(Nair & Hinton, 2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) revived deep learning by solving vanishing gradients in deep CNNs:

- Gradient is exactly 1 in the active region (no saturation).
- Computationally trivial: a single $\max$.
- Sparse activations (~half are zero): biological intuition + computational efficiency.

But ReLU has the **dying ReLU problem**: a neuron stuck at $z < 0$ has gradient 0 forever and never recovers.

**GELU** [(Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415) and **swish / SiLU** [(Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941) are smooth versions of ReLU that have non-zero gradient everywhere. Empirically improve transformer training over ReLU; the gain is small (~1% perplexity) but consistent.

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
- **Forgetting GELU has two forms.** Exact (CDF-based) and approximate (tanh-based polynomial). They give visibly different outputs near 0; check your framework.

## Related

- [Weight initialization](/reference/weight-initialization/). Initialization is activation-dependent.
- [Exploding and vanishing gradients](/reference/exploding-vanishing-gradients/). What motivates ReLU and gated activations.
