---
title: "Vision transformers (ViT)"
description: "Apply a standard transformer to a sequence of image patches. Beats CNNs at scale; the dominant backbone for foundation vision models in 2026."
date: "2025-12-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **Vision Transformer** [(Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) splits an image into fixed-size patches (e.g., $16 \times 16$ pixels), linearly embeds each patch into a vector, adds positional embeddings, and processes the resulting sequence with a standard transformer encoder. A learned `[CLS]` token (or global average pool over patch tokens) feeds the classifier.

ViT showed that **transformers can match or beat CNNs** on image classification given enough training data. It triggered the convergence of vision and language architectures: same building block, same training approach, same scaling laws. Modern vision foundation models (DINOv2, SAM, CLIP image encoder, MaskedAutoencoder, EVA, BEiT) all use ViT backbones.

ViT also inherits transformer-specific advantages: easier multi-modal fusion (concat image + text tokens), simpler architecture (no convolution-specific kernels), better scaling with data and compute.

## The architecture

Input: image $x \in \mathbb{R}^{H \times W \times 3}$.

1. **Patchify**: split into $N = HW/P^2$ patches of size $P \times P$ (typically $P = 16$). Reshape each to a vector of length $3P^2$.
2. **Linear projection**: project each patch to dimension $d$. Result: sequence of $N$ embeddings.
3. **Prepend `[CLS]` token**: a learned embedding that aggregates global information.
4. **Add positional embeddings**: learned 1D positions; some variants use 2D.
5. **Transformer encoder**: $L$ layers of self-attention + FFN with LayerNorm and residuals.
6. **Classification head**: linear layer on the `[CLS]` embedding (or global average pool over patch embeddings).

That's it. No convolutions, no inductive biases beyond patchification.

## Patch size matters

| Patch | Sequence length (224×224) | Compute | Spatial resolution |
|-------|--------------------------|---------|-------------------|
| 32 | 49 | low | coarse |
| 16 | 196 | medium (most common) | medium |
| 8 | 784 | high | fine |
| 4 | 3136 | very high | very fine |

Smaller patches → longer sequence → more compute (attention is $O(N^2)$) → finer detail. ViT-B/16 (base, patch 16) is the workhorse.

## Sizes

ViT was released in three sizes [(Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929):

| Model | Layers | Width | Heads | Params |
|-------|--------|-------|-------|--------|
| ViT-B (Base) | 12 | 768 | 12 | 86M |
| ViT-L (Large) | 24 | 1024 | 16 | 307M |
| ViT-H (Huge) | 32 | 1280 | 16 | 632M |

Modern foundation vision models scale to >1B parameters (EVA-CLIP, DINOv2-G, ViT-22B).

## ViT vs. CNN: data and compute

The original ViT result: with **little data** (ImageNet-1k), CNNs win. With **large pretraining data** (ImageNet-21k, JFT-300M), ViT matches or beats CNNs. With **lots more data**, the gap grows.

Why: CNNs encode strong inductive biases (translation equivariance, locality) that act as data efficiency. ViT has none of those. It must learn them from data. But is more expressive once enough data is available.

In 2026: ViT dominates large-scale vision pretraining. CNNs (ConvNeXt) remain competitive on standard benchmarks at matched compute. For small-data transfer learning, both work.

## Pretraining strategies

ViT became the default vision backbone partly because of self-supervised pretraining methods that work well with it:

- **Contrastive (CLIP, SigLIP)**: align image and caption embeddings. Produces zero-shot classifiers.
- **Masked image modeling (MAE, BEiT, SimMIM)**: mask 75% of patches, reconstruct from the visible ones. Produces strong representations for downstream fine-tuning.
- **DINO / DINOv2**: self-distillation. State-of-the-art representations for dense and global tasks.

## Variants

- **DeiT** [(Touvron 2020)](https://arxiv.org/abs/2012.12877): training recipe to match CNN data efficiency without extra data.
- **Swin Transformer** [(Liu 2021)](https://arxiv.org/abs/2103.14030): hierarchical with shifted-window attention; CNN-like inductive biases.
- **PVT, MViT**: pyramid structures for dense prediction (segmentation, detection).
- **Hybrid CNN + ViT**: convolutions early (low-level features), transformer later. Used in some detection / segmentation pipelines.

## Common pitfalls

- **Training ViT from scratch on small data.** It will lose to a ResNet. Pretrain on ImageNet-21k or use DINOv2 weights.
- **Using `[CLS]` vs. global pool inconsistently.** Both work; pick one based on the pretraining recipe (DINO uses global pool, original ViT uses CLS).
- **Forgetting positional embeddings.** Without them, ViT is permutation-invariant. Performance collapses.
- **Treating patch size as a free hyperparameter.** Smaller patches massively increase compute; doesn't always pay off.
- **Comparing ViT-B against ResNet-50 on params alone.** Different operations, different cost; compare on FLOPs and accuracy.

## Related

- [Attention mechanism](/concepts/attention-mechanism/). The core operation.
- [Transformer architecture](/concepts/transformer-architecture/). Block structure.
- [CNN architecture](/concepts/cnn-architecture/). Alternative paradigm.
