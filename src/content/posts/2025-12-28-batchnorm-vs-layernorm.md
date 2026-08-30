---
title: "BatchNorm vs LayerNorm (and the transformer wrinkle)"
description: "BatchNorm and LayerNorm normalize different axes and behave differently during training. The axis and train/eval behavior determine which one fits."
date: "2025-12-28"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

A normalization layer is parameterized by **(a) which dimensions you normalize across** and **(b) which dimensions get learnable scale/shift parameters**. BatchNorm and LayerNorm differ on both axes, and the way LayerNorm is used in transformers is *not* the same as how it was originally specified for sequence models.

## The general form

Every normalization layer does:

```
y = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
```

The whole question is: **over which axes are &mu; and &sigma; computed?** And: **which axes do &gamma; and &beta; have?**

For a 4-D activation tensor of shape `(N, C, H, W)` (batch, channel, height, width):

| Norm | Stats computed across | &gamma;, &beta; shape |
|---|---|---|
| **BatchNorm (BN)** | N, H, W (per channel) | C |
| **LayerNorm in CNNs (LN-CNN)** | C, H, W (per sample) | C, H, W |
| **LayerNorm in transformers (LN-TX)** | D (per token) | D |
| **InstanceNorm** | H, W (per sample, per channel) | C |
| **GroupNorm** | G groups of C/G channels, H, W | C |

<!-- visual:normalization-shared-statistics -->
<figure class="learning-figure" aria-labelledby="normalization-axes-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="normalization-axes-title">Color groups values that share one mean and variance.</p>
	<div class="visual-grid--two">
		<section class="visual-panel" aria-labelledby="batchnorm-panel-title">
			<h4 id="batchnorm-panel-title">BatchNorm on (N, C, H, W)</h4>
			<p>Each column is one channel across samples. Every square also represents all H×W positions.</p>
			<div class="norm-axis-key"><strong>columns:</strong> C0 · C1 · C2 <strong>rows:</strong> sample N</div>
			<div class="norm-matrix norm-matrix--batch" role="img" aria-label="Four samples by three channels. Values in each channel column share one mean and variance across batch and spatial positions.">
				<span class="norm-cell norm-c0">N0 C0</span><span class="norm-cell norm-c1">N0 C1</span><span class="norm-cell norm-c2">N0 C2</span>
				<span class="norm-cell norm-c0">N1 C0</span><span class="norm-cell norm-c1">N1 C1</span><span class="norm-cell norm-c2">N1 C2</span>
				<span class="norm-cell norm-c0">N2 C0</span><span class="norm-cell norm-c1">N2 C1</span><span class="norm-cell norm-c2">N2 C2</span>
				<span class="norm-cell norm-c0">N3 C0</span><span class="norm-cell norm-c1">N3 C1</span><span class="norm-cell norm-c2">N3 C2</span>
			</div>
			<p class="norm-brace">One (μ<sub>c</sub>, σ<sub>c</sub>) per colored channel, shared across N, H, and W.</p>
			<p class="norm-behavior">Train: batch statistics. Eval: stored running statistics.</p>
		</section>
		<section class="visual-panel" aria-labelledby="layernorm-panel-title">
			<h4 id="layernorm-panel-title">LayerNorm on transformer (B, T, D)</h4>
			<p>Each row is one token. Its embedding dimensions share statistics only with each other.</p>
			<div class="norm-axis-key"><strong>columns:</strong> D0…D5 <strong>rows:</strong> token (b, t)</div>
			<div class="norm-matrix norm-matrix--layer" role="img" aria-label="Four token rows by six embedding dimensions. Each token row computes its own mean and variance across embedding dimensions, independently of other tokens and samples.">
				<span class="norm-cell norm-row-a">D0</span><span class="norm-cell norm-row-a">D1</span><span class="norm-cell norm-row-a">D2</span><span class="norm-cell norm-row-a">D3</span><span class="norm-cell norm-row-a">D4</span><span class="norm-cell norm-row-a">D5</span>
				<span class="norm-cell norm-row-b">D0</span><span class="norm-cell norm-row-b">D1</span><span class="norm-cell norm-row-b">D2</span><span class="norm-cell norm-row-b">D3</span><span class="norm-cell norm-row-b">D4</span><span class="norm-cell norm-row-b">D5</span>
				<span class="norm-cell norm-row-c">D0</span><span class="norm-cell norm-row-c">D1</span><span class="norm-cell norm-row-c">D2</span><span class="norm-cell norm-row-c">D3</span><span class="norm-cell norm-row-c">D4</span><span class="norm-cell norm-row-c">D5</span>
				<span class="norm-cell norm-row-d">D0</span><span class="norm-cell norm-row-d">D1</span><span class="norm-cell norm-row-d">D2</span><span class="norm-cell norm-row-d">D3</span><span class="norm-cell norm-row-d">D4</span><span class="norm-cell norm-row-d">D5</span>
			</div>
			<p class="norm-brace">One (μ<sub>b,t</sub>, σ<sub>b,t</sub>) per colored token row, computed across D.</p>
			<p class="norm-behavior">Train and eval: the same per-token computation.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> BatchNorm shares statistics down a channel column and therefore couples examples. Transformer LayerNorm shares statistics across one token row and never mixes tokens or samples.</figcaption>
</figure>

**LayerNorm normalizes across all features of a sample.** In CNNs that means C&times;H&times;W. In transformers that means only D, the embedding dimension. These are different normalization axes despite the shared name.

**BatchNorm:**
- Couples samples in a batch together (statistics are computed across N).
- Behaves differently at train and eval time (eval uses running averages).
- Breaks at small batch sizes (statistics get noisy when N &lt; ~16).
- Breaks for sequence models with variable-length inputs and packed batches.
- Strong regularizer (the noise from batch statistics is the regularization).

**LayerNorm:**
- Each sample is normalized independently. No batch coupling.
- Same behavior at train and eval.
- Works at any batch size, including 1.
- Works for sequence models with variable lengths.
- Much weaker regularizer; you usually need additional regularization.

This is why CNNs adopted BN and transformers adopted LN. The choice was not stylistic; it was forced by the structural properties of each architecture.

## What an interviewer expects you to say

If asked "BatchNorm vs LayerNorm":

1. State the general form (subtract mean, divide by std, scale-shift).
2. Specify which axes the statistics are computed over for each.
3. Mention train/eval mode difference for BN; absence for LN.
4. Explain why each is used where it is, CNNs (BN) vs transformers (LN), and why.
5. Bonus: mention the LN-CNN vs LN-Transformer wrinkle (different normalization axes despite the same name).

Discussing pre-norm vs post-norm transformers and RMSNorm marks senior-level knowledge.

## The transformer-specific wrinkle most people miss

In a transformer, an activation has shape `(B, T, D)` where T = sequence length, D = embedding dim.

LayerNorm normalizes **only across the D dimension**: per (b, t). It does *not* normalize across T (that would mix tokens) and *not* across B (that would couple samples like BN does).

This is critical for two reasons:
1. **Variable-length sequences work natively.** Each token's statistics depend only on its own D values; padding doesn't pollute them.
2. **Sequence packing works.** When you concatenate multiple short sequences into a packed batch, LN doesn't care, statistics are per-token. (BN would catastrophically fail here; the statistics would mix examples.)

The strongest answer to "why don't transformers use BatchNorm": it doesn't just work worse, BN actively breaks what makes transformer training tractable.

## RMSNorm: the modern variant

Most production LLMs from 2023 onward use **RMSNorm** instead of LayerNorm:

```
y = gamma * x / sqrt(mean(x^2) + eps)
```

The difference: skip the mean-subtraction, just normalize by the root mean square. Two consequences:
- ~7-15% faster (one fewer reduction).
- Empirically equivalent or better quality.

In high dimensions, random projection means are near-zero anyway, so subtraction is mostly noise. Empirically validated.

RMSNorm knowledge is expected in 2026 transformer discussions.

## Pre-norm vs post-norm

Where you put the LN matters:

- **Post-norm (original transformer)**: `y = LN(x + Sublayer(x))`. Norm is *after* the residual. Hard to train at depth without a careful warm-up schedule.
- **Pre-norm (modern default)**: `y = x + Sublayer(LN(x))`. Norm is *before* the sublayer. Easier to train, more stable at scale, but slightly worse final quality at small scales.

Almost all modern LLMs use pre-norm. If you write a transformer and put the LN after the residual, you're inviting training instability.

## Common confusions

- **"LayerNorm normalizes across the layer."** No, it normalizes across the *features within a sample*. The "layer" in the name is historical and misleading.
- **"BatchNorm and LayerNorm are interchangeable."** They have very different inductive biases; swapping them is a real architectural change, not a stylistic one.
- **"BatchNorm regularizes by reducing internal covariate shift."** This was the original justification; subsequent papers showed it's actually wrong. BN works for other reasons (smoother loss landscape, implicit regularization from batch noise). Don't say "internal covariate shift" in a 2026 interview unless you're prepared to immediately note that the explanation is contested.
- **"LayerNorm is just BatchNorm with batch size 1."** No. The axes being normalized over are different.

## Why interviewers care

This question tests whether you understand:
1. Activation tensor shapes and axes.
2. The relationship between architecture and normalization choice.
3. Train/eval mode subtleties.
4. Whether your knowledge has been updated since 2018.

Easy to fumble; easy to ace. Worth memorizing.

---

*Related: [FlashAttention](/concepts/flashattention/) for the other transformer-specific optimization that's often paired with this in questions.*
