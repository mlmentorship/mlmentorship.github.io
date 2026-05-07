---
title: "Why is softmax + cross-entropy the right pairing?"
description: "The gradient simplifies to (p - y), and that's not a coincidence. The senior answer derives this and connects to GLMs and numerical stability."
date: "2026-04-26"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: ML breadth, math-screen, and LLM internals interviews.*

The L4 candidate states the pairing. The L6 candidate derives the gradient simplification, explains the GLM connection, and discusses why frameworks compute the joint operation rather than the two separately.

## The setup

C-class classification. Logits `z = (z_1, ..., z_C)`. Softmax produces probabilities:

```
p_i = exp(z_i) / sum_j exp(z_j)
```

Cross-entropy with one-hot label `y`:

```
L = -sum_i y_i * log p_i = -log p_y
```

(where `y` is the index of the true class).

## The gradient simplification

Compute `dL / dz_k`:

```
dL / dz_k = -d log p_y / dz_k
         = -(1 / p_y) * dp_y / dz_k
```

Two cases for `dp_y / dz_k`:
- If `k == y`: `dp_y / dz_k = p_y * (1 - p_y)`.
- If `k != y`: `dp_y / dz_k = -p_y * p_k`.

Substituting:
- `dL / dz_y = -(1 / p_y) * p_y * (1 - p_y) = -(1 - p_y) = p_y - 1`
- `dL / dz_k = -(1 / p_y) * (-p_y * p_k) = p_k`

Combining: `dL / dz_k = p_k - y_k` where `y_k = 1` for `k = y`, `0` otherwise.

The full gradient is `p - y` (predicted probabilities minus the one-hot true label). Three lines of algebra; the cleanest gradient in deep learning.

## Why this matters

> "Three reasons the joint operation is preferred:
>
> **1. Numerical stability.** Computing softmax then cross-entropy separately involves taking `log(exp(...))`, which can overflow or underflow. The joint operation uses log-sum-exp:
>
> ```
> log p_y = z_y - log sum_j exp(z_j) = z_y - max_j z_j - log sum_j exp(z_j - max_j z_j)
> ```
>
> Subtracting `max_j z_j` keeps the largest exponentiated argument at zero, avoiding overflow. Frameworks (PyTorch's `nn.CrossEntropyLoss`) accept logits directly and apply this internally.
>
> **2. Computational efficiency.** The joint operation skips computing the explicit probabilities (since the gradient `p - y` only needs `p`, computed on demand). Saves memory and a few flops.
>
> **3. The gradient is exact and stable.** The `p - y` form has bounded magnitude (each element is in [-1, 1]), so gradients don't explode at the loss layer."

## The L6 connection: GLM

> "Softmax + cross-entropy is the multiclass generalization of sigmoid + binary cross-entropy, both of which are GLMs under their canonical link functions. The gradient simplification `(predicted - true) * input` is a property of *all* canonical-link GLMs, not just classification. Linear regression with MSE has the same gradient form (because MSE on a Gaussian noise model is the GLM with identity link).
>
> This explains why modern deep nets almost universally use sigmoid + BCE for binary, softmax + CE for multiclass: not just convention, but the gradient and stability properties make these the natural choices."

## Tells that get you a strong-hire vote

- You **derive the gradient** cleanly.
- You name the **log-sum-exp trick** for numerical stability.
- You connect to **GLMs and canonical links**.
- You explain why frameworks **fuse the operations**.

## Tells that get you down-leveled

- "It just works" without derivation.
- Computing softmax explicitly in code (in real systems, you should pass logits to the loss function).
- No mention of numerical stability.
- Confusion about which axis softmax operates on.

## Common follow-up

"What's wrong with using MSE for classification?"

The L6 answer:

> "Two related problems. (1) Vanishing gradients on confident-wrong predictions: MSE gradient under sigmoid is proportional to `(p - y) * p * (1 - p)`. When the model is very confident and wrong (`p ≈ 1` for the wrong class), the `p * (1 - p)` term vanishes; the model can't learn its way out. Cross-entropy's gradient is `p - y` directly, which stays large precisely when the model is most wrong. (2) MSE assumes Gaussian noise; classification labels are categorical. MLE under the wrong noise model gives the wrong objective. Cross-entropy is MLE under the right (categorical) noise model."

---

*Related reference: [Cross-entropy and softmax](/reference/cross-entropy-softmax/), [Derive logistic regression from MLE](/interviews/derive-logistic-regression/), [How to choose a loss function](/interviews/how-to-choose-loss-function/).*
