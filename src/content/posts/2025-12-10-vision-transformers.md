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

<!-- visual:vision-patches-to-positioned-tokens -->
<figure class="learning-figure plot-panel" aria-labelledby="vit-patch-sequence-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="vit-patch-sequence-title">How does a two-dimensional image become an ordered transformer sequence?</p>
	<svg viewBox="0 0 360 640" role="img" aria-labelledby="vit-patch-svg-title vit-patch-svg-desc">
		<title id="vit-patch-svg-title">Image patches become positioned Vision Transformer tokens</title>
		<desc id="vit-patch-svg-desc">A toy four by four RGB image is divided into four non-overlapping two by two patches in row-major order: P1 top-left, P2 top-right, P3 bottom-left, and P4 bottom-right. Each patch contains twelve channel values and is flattened and linearly projected to one width-d embedding. A class token is prepended to embeddings e1 through e4. Learned position vectors p0 through p4 are added slot by slot, producing five positioned tokens z0 through z4 for a standard transformer encoder. The encoder's z0 class-token output feeds the classifier.</desc>
		<defs><marker id="vit-patch-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path class="viz-arrow-forward" d="M0 0L8 4L0 8Z"></path></marker></defs>
		<text class="viz-axis-label" x="14" y="20">1 · CUT A 2D GRID INTO NON-OVERLAPPING PATCHES</text>
		<rect class="viz-node viz-node--input" x="116" y="38" width="128" height="128" rx="3"></rect>
		<path class="viz-gridline" d="M148 38V166M180 38V166M212 38V166M116 70H244M116 102H244M116 134H244"></path>
		<path d="M180 38V166M116 102H244" style="fill:none;stroke:var(--c-text-soft);stroke-width:2.5"></path>
		<rect class="viz-node viz-node--focus" x="116" y="102" width="64" height="64"></rect>
		<text class="viz-callout" x="148" y="74" text-anchor="middle">P1 · r1 c1</text>
		<text class="viz-callout" x="212" y="74" text-anchor="middle">P2 · r1 c2</text>
		<text class="viz-callout" x="148" y="138" text-anchor="middle">P3 · r2 c1</text>
		<text class="viz-callout" x="212" y="138" text-anchor="middle">P4 · r2 c2</text>
		<text class="viz-label" x="180" y="185" text-anchor="middle">toy image 4 × 4 × 3 · patch size 2 × 2</text>
		<path d="M180 194V220" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vit-patch-arrow)"></path>
		<text class="viz-axis-label" x="14" y="238">2 · FLATTEN CONTENT, THEN PROJECT TO WIDTH d</text>
		<rect class="viz-node" x="22" y="252" width="316" height="76" rx="4"></rect>
		<text class="viz-node-label" x="180" y="275">each patch P<tspan baseline-shift="sub" font-size="10">i</tspan> becomes one embedding e<tspan baseline-shift="sub" font-size="10">i</tspan></text>
		<text class="viz-node-value" x="180" y="297">2 × 2 × 3 = 12 values → flatten → linear W<tspan baseline-shift="sub" font-size="8">E</tspan> → ℝ<tspan baseline-shift="super" font-size="8">d</tspan></text>
		<text class="viz-callout" x="180" y="317" text-anchor="middle">row-major order: P1, P2, P3, P4 → e1, e2, e3, e4</text>
		<path d="M180 336V359" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vit-patch-arrow)"></path>
		<text class="viz-axis-label" x="14" y="378">3 · PREPEND CLS, THEN ADD ONE POSITION VECTOR PER SLOT</text>
		<text class="viz-label" x="15" y="409">content</text>
		<rect class="viz-node" x="66" y="389" width="52" height="31" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text class="viz-callout" x="92" y="409" text-anchor="middle">[CLS]</text>
		<rect class="viz-node viz-node--input" x="124" y="389" width="52" height="31" rx="3"></rect><text class="viz-callout" x="150" y="409" text-anchor="middle">e1</text>
		<rect class="viz-node viz-node--input" x="182" y="389" width="52" height="31" rx="3"></rect><text class="viz-callout" x="208" y="409" text-anchor="middle">e2</text>
		<rect class="viz-node viz-node--focus" x="240" y="389" width="52" height="31" rx="3"></rect><text class="viz-callout" x="266" y="409" text-anchor="middle">e3</text>
		<rect class="viz-node viz-node--input" x="298" y="389" width="52" height="31" rx="3"></rect><text class="viz-callout" x="324" y="409" text-anchor="middle">e4</text>
		<text class="viz-callout" x="92" y="438" text-anchor="middle">+</text><text class="viz-callout" x="150" y="438" text-anchor="middle">+</text><text class="viz-callout" x="208" y="438" text-anchor="middle">+</text><text class="viz-callout" x="266" y="438" text-anchor="middle">+</text><text class="viz-callout" x="324" y="438" text-anchor="middle">+</text>
		<text class="viz-label" x="15" y="469">position</text>
		<rect class="viz-node" x="66" y="449" width="52" height="31" rx="3"></rect><text class="viz-callout" x="92" y="469" text-anchor="middle">p0</text>
		<rect class="viz-node" x="124" y="449" width="52" height="31" rx="3"></rect><text class="viz-callout" x="150" y="469" text-anchor="middle">p1</text>
		<rect class="viz-node" x="182" y="449" width="52" height="31" rx="3"></rect><text class="viz-callout" x="208" y="469" text-anchor="middle">p2</text>
		<rect class="viz-node viz-node--focus" x="240" y="449" width="52" height="31" rx="3"></rect><text class="viz-callout" x="266" y="469" text-anchor="middle">p3</text>
		<rect class="viz-node" x="298" y="449" width="52" height="31" rx="3"></rect><text class="viz-callout" x="324" y="469" text-anchor="middle">p4</text>
		<text class="viz-callout" x="92" y="499" text-anchor="middle">=</text><text class="viz-callout" x="150" y="499" text-anchor="middle">=</text><text class="viz-callout" x="208" y="499" text-anchor="middle">=</text><text class="viz-callout" x="266" y="499" text-anchor="middle">=</text><text class="viz-callout" x="324" y="499" text-anchor="middle">=</text>
		<text class="viz-label" x="15" y="530">encoder</text>
		<rect class="viz-node" x="66" y="510" width="52" height="31" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text class="viz-callout" x="92" y="530" text-anchor="middle">z0</text>
		<rect class="viz-node" x="124" y="510" width="52" height="31" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text class="viz-callout" x="150" y="530" text-anchor="middle">z1</text>
		<rect class="viz-node" x="182" y="510" width="52" height="31" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text class="viz-callout" x="208" y="530" text-anchor="middle">z2</text>
		<rect class="viz-node viz-node--focus" x="240" y="510" width="52" height="31" rx="3"></rect><text class="viz-callout" x="266" y="530" text-anchor="middle">z3</text>
		<rect class="viz-node" x="298" y="510" width="52" height="31" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text class="viz-callout" x="324" y="530" text-anchor="middle">z4</text>
		<path d="M180 550V572" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#vit-patch-arrow)"></path>
		<rect class="viz-node viz-node--output" x="48" y="578" width="264" height="46" rx="4"></rect>
		<text class="viz-node-label" x="180" y="597">standard transformer encoder</text>
		<text class="viz-node-value" x="180" y="614">five tokens in · z0 output → classifier</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow highlighted P3: its pixels are flattened and projected into e3, then the learned vector p3 marks that embedding as the bottom-left patch before self-attention. The same slot-by-slot addition applies to every patch and to `[CLS]`. Patchification changes the layout, not the transformer block: the encoder receives an ordinary five-token sequence in this toy example. Original schematic checked against <a href="https://arxiv.org/abs/2010.11929">Dosovitskiy et al.</a> and the <a href="https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py">Google Research reference implementation</a>.</figcaption>
</figure>

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
