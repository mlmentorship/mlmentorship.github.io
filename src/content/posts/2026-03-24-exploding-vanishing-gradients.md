---
title: "Exploding and vanishing gradients"
description: "Why deep networks were untrainable before residuals, normalization, and ReLU. The math of gradient magnitudes through depth and the standard fixes."
date: "2026-03-24"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Gradients flowing back through a deep network multiply many Jacobians together. If the average per-layer Jacobian norm is $> 1$, gradient magnitudes grow exponentially with depth (**exploding**); if $< 1$, they shrink to zero (**vanishing**). Either failure mode prevents the early layers from learning.

## Why it matters

This was the central obstacle to training deep networks before ~2014. The standard fixes. Careful initialization, normalization layers, residual connections, ReLU activations, gradient clipping. Exist primarily to control gradient magnitudes through depth. Knowing the failure mode and the fixes is core senior-level material.

## The math

For a deep net $L \circ f_n \circ f_{n-1} \circ \dots \circ f_1$, the gradient w.r.t. $f_1$ is:

$$
\nabla_{f_1} L = \nabla_{f_n} L \cdot J_{f_n} \cdot J_{f_{n-1}} \cdot \dots \cdot J_{f_2}.
$$

Each $J_{f_i} = \partial f_i / \partial f_{i-1}$. The gradient norm scales roughly as the product of the per-layer Jacobian operator norms. With $n$ layers and average norm $\rho$:

$$
\|\nabla_{f_1} L\| \sim \rho^{n-1}.
$$

For $\rho = 1.1$ and $n = 50$: factor of ~117 (gradients explode). For $\rho = 0.9$ and $n = 50$: factor of ~0.005 (gradients vanish).

## Where it shows up

| Architecture | Failure mode |
|--------------|--------------|
| Deep MLP with sigmoid activations | Vanishing (sigmoid derivative ≤ 0.25, multiplies away) |
| RNNs with tanh / sigmoid (long sequences) | Both: gradient through time multiplies $\partial h_{t+1}/\partial h_t$ many times |
| Deep CNN without normalization | Vanishing in early layers |
| Transformers without LayerNorm + residuals | Both: hard to train past 6–12 layers |

## The fixes (and what they actually do)

### Weight initialization
Scale initial weights so that per-layer activation variance is preserved on the forward pass and gradient variance is preserved on the backward. **Kaiming / He init** (for ReLU) and **Xavier / Glorot init** (for tanh) achieve this. Without it, gradients vanish or explode at step 0. See [weight initialization](/concepts/weight-initialization/).

### Non-saturating activations
**ReLU** has gradient exactly 1 in the active region; doesn't shrink gradients through depth (unlike sigmoid / tanh which max out at 0.25). Modern alternatives: GELU, swish (smooth, non-saturating).

### Normalization
**BatchNorm**, **LayerNorm**, **RMSNorm** stabilize activation distributions across layers, indirectly stabilizing gradient magnitudes. LayerNorm in transformers is essential. Without it, deep transformers don't train.

### Residual connections
**Skip connections** ($f(x) + x$) provide a "highway" for gradients to flow back without being attenuated through every layer's Jacobian. Enabled deep ResNets (152+ layers) and made deep transformers practical. The gradient now contains an "identity" term that bypasses each block.

### Gradient clipping
Cap the gradient norm at a fixed threshold ($c = 1.0$ standard for transformers). Doesn't prevent exploding gradients structurally, but keeps any single optimizer step from causing divergence. See [gradient clipping](/concepts/gradient-clipping/).

### Better optimizers
Adam-family optimizers normalize per-parameter gradients by their running variance, partially counteracting magnitude differences across layers.

## Diagnostics

If your deep network won't train:

- Log per-layer **gradient norm** during training. Vanishing: early layers have norm $< 10^{-6}$. Exploding: late layers have norm $> 10^4$.
- Log per-layer **activation magnitudes**. Should be roughly constant across depth; collapsing or exploding indicates trouble.
- Single-batch overfit. If a deep net can't memorize one batch, suspect optimization pathology.

## Common pitfalls

- **Using sigmoid in deep MLP hidden layers.** Vanishing gradients. Use ReLU or GELU.
- **Stacking transformer blocks without LayerNorm.** Deep transformers won't train without it (or RMSNorm).
- **Using PyTorch default `nn.Linear` init for transformers.** Default Kaiming-uniform is wrong scale for typical transformer FFN; many implementations override to $\mathcal{N}(0, 0.02^2)$.
- **Treating gradient clipping as a substitute for proper architecture.** Clipping bounds individual steps but doesn't fix structural vanishing.
- **Assuming residual = problem solved.** Residual connections help enormously but don't substitute for normalization or sensible init.
