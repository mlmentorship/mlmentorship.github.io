---
title: "Residual connections"
description: "Add the input of a block to its output. Lets gradients flow unimpeded through depth and made networks deeper than 30 layers practical for the first time."
date: "2025-11-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A **residual connection** (skip connection) makes a block compute $y = x + f(x)$ instead of $y = f(x)$, so that the block's output adds to its input rather than replacing it. Introduced by ResNet [(He et al., 2015)](https://arxiv.org/abs/1512.03385) and ubiquitous in every modern deep architecture.

## Why it matters

Pre-ResNet (2014), networks past ~20 layers showed *worse* training accuracy than shallower networks. Not from overfitting but from optimization pathology. Residual connections solved this and made 152-layer ResNets routine, then 1000-layer networks (with normalization) feasible. Every modern architecture. ResNets, transformers, U-Nets, diffusion models, MLP-Mixers. Uses residuals.

## The mechanism

A residual block:

```
y = x + f(x)
```

where $f$ is the "residual function". Typically (Conv → BN → ReLU → Conv → BN) for ResNet or (LayerNorm → Attn → Linear) for transformer attention.

The forward pass is trivial. The interesting effect is on gradients:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left( I + \frac{\partial f}{\partial x} \right).
$$

The identity matrix $I$ in the parenthesis is the "residual gradient highway". Gradients flow back through the identity term without being multiplied by Jacobians of $f$. Even if $f$ is poorly conditioned or near-zero gradient, the identity ensures gradient signal reaches earlier layers.

## Why it works (intuitions)

Three complementary explanations:

1. **Easier to learn the identity.** If the optimal $f$ is near zero (i.e., the layer is unhelpful), the network can simply set $f \to 0$ and the block becomes the identity. Without the residual, learning the identity through a deep stack of ReLU+linear is hard.
2. **Gradient highway.** As above; identity term in the backward pass prevents vanishing.
3. **Implicit ensemble** [(Veit et al., 2016)](https://arxiv.org/abs/1605.06431): a depth-$N$ ResNet acts like an ensemble of $2^N$ paths of varying depth, with the shallow paths providing strong learning signal.

## Pre-norm vs. post-norm in transformers

Two arrangements of the residual + normalization in transformer blocks:

- **Post-norm** (original transformer): $y = \text{LayerNorm}(x + f(x))$. Used in original Vaswani et al. (2017).
- **Pre-norm**: $y = x + f(\text{LayerNorm}(x))$. Used in GPT-2/3, Llama, Mistral, every modern decoder.

Pre-norm is much more stable to train at depth; the residual stream is never normalized, so gradient magnitudes stay bounded. Post-norm requires careful warmup. Pre-norm is the default in 2026.

## Bottleneck blocks

For very deep networks (ResNet-50/101/152), the residual block is replaced with a **bottleneck**:

```
y = x + Conv1x1 ↓ → Conv3x3 → Conv1x1 ↑ (x)
```

Reduce channels with $1 \times 1$ conv, do the expensive $3 \times 3$ at low channel count, expand back. Cuts compute roughly 4× per block at similar accuracy.

## Where residuals show up

| Architecture | Where |
|--------------|-------|
| ResNet, ResNeXt, Wide ResNet | Every block |
| U-Net | Across encoder-decoder paths (long skips) |
| Transformer (encoder & decoder) | Around attention and FFN sub-blocks |
| Diffusion U-Nets | Both within blocks and across encoder-decoder |
| MLP-Mixer, ConvNeXt | Block residuals |

Almost no modern architecture omits residuals.

## Common pitfalls

- **Adding residual through dimension change.** $x + f(x)$ requires matching shapes. When channel count changes, project $x$ with a $1 \times 1$ conv (ResNet) or linear (transformer with embedding-dim mismatch).
- **Putting normalization before vs. after the residual.** Pre-norm vs. post-norm have very different training dynamics; pre-norm is the safe choice in 2026.
- **Skipping the residual scaling in deep stacks.** Some recipes scale the residual contribution by $1/\sqrt{N}$ for $N$ layers (GPT-2 style); useful for very deep stacks.
- **Treating residuals as "free."** They cost a small amount of memory (need to keep $x$ around for the addition) and contribute to activation memory.

## Related

- [Exploding and vanishing gradients](/concepts/exploding-vanishing-gradients/). The problem residuals solve.
- [Transformer architecture](/concepts/transformer-architecture/). The canonical user.
